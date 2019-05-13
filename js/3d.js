(function() {

  function createShader(gl, source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader) + source);
    }
    return shader;
  }

  function createProgramFromSource(gl, vertexShaderSource, fragmentShaderSource) {
    const program = gl.createProgram();
    gl.attachShader(program, createShader(gl, vertexShaderSource, gl.VERTEX_SHADER));
    gl.attachShader(program, createShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
    }
    return program;
  }

  function getUniformLocations(gl, program, keys) {
    const locations = {};
    keys.forEach(key => {
        locations[key] = gl.getUniformLocation(program, key);
    });
    return locations;
  }

  function createTexture(gl, width, height, internalFormat, format, type) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return texture;
  }

  function createVelocityFramebuffer(gl, width, height) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const velocityTexture = createTexture(gl, width, height, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, velocityTexture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      velocityTexture: velocityTexture
    };
  }

  function createPressureFramebuffer(gl, width, height) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const pressureTexture = createTexture(gl, width, height, gl.R32F, gl.RED, gl.FLOAT);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pressureTexture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      pressureTexture: pressureTexture
    };
  }

  function createSmokeFramebuffer(gl, width, height) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const smokeTexture = createTexture(gl, width, height, gl.RG32F, gl.RG, gl.FLOAT);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, smokeTexture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      smokeTexture: smokeTexture
    };
  }

  function setUniformTexture(gl, index, texture, location) {
    gl.activeTexture(gl.TEXTURE0 + index);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(location, index);
  }

  const FILL_VIEWPORT_VERTEX_SHADER_SOURCE =
`#version 300 es

const vec3[4] POSITIONS = vec3[](
  vec3(-1.0, -1.0, 0.0),
  vec3(1.0, -1.0, 0.0),
  vec3(-1.0, 1.0, 0.0),
  vec3(1.0, 1.0, 0.0)
);

const int[6] INDICES = int[](
  0, 1, 2,
  3, 2, 1
);

void main(void) {
  vec3 position = POSITIONS[INDICES[gl_VertexID]];
  gl_Position = vec4(position, 1.0);
}
`;

  const INITIALIZE_VELOCITY_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec3 o_velocity;

void main(void) {
  o_velocity = vec3(0.0, 0.01, 0.0);
}
`;

  const INITIALIZE_SMOKE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec2 o_smoke;

uniform int u_cellNum;
uniform int u_cellTextureSize;
uniform ivec3 u_resolution;
uniform float u_gridSpacing;
uniform vec3 u_simulationSpace;

#define AMBIENT_TEMPERATURE 273.0

int convertCoordToCellId(ivec2 coord) {
  return coord.x + coord.y * u_cellTextureSize;
}

ivec3 convertCellIdToCellIndex(int cellId) {
  int z = cellId / (u_resolution.x * u_resolution.y);
  int y = (cellId % (u_resolution.x * u_resolution.y)) / u_resolution.x;
  int x = cellId % u_resolution.x;
  return ivec3(x, y, z);
}

vec3 convertCellIdToPosition(int cellId) {
  ivec3 cellIndex = convertCellIdToCellIndex(cellId);
  return (vec3(cellIndex) + 0.5) * u_gridSpacing;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int cellId = convertCoordToCellId(coord);
  if (cellId >= u_cellNum) {
    return;
  }
  vec3 position = convertCellIdToPosition(cellId);
  float density = 0.0;
  float temperature = AMBIENT_TEMPERATURE;
  o_smoke = vec2(density, temperature);
}
`;

  const ADD_BUOYANCY_FORCE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec3 o_velocity;

uniform int u_cellNum;
uniform int u_cellTextureSize;
uniform sampler2D u_velocityTexture;
uniform sampler2D u_smokeTexture;
uniform float u_deltaTime;
uniform float u_densityScale;
uniform float u_temperatureScale;

#define GRAVITY vec3(0.0, -9.8, 0.0)
#define AMBIENT_TEMPERATURE 273.0

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int cellId = coord.x + coord.y * u_cellTextureSize;
  if (cellId >= u_cellNum) {
    return;
  }
  vec3 velocity = texelFetch(u_velocityTexture, coord, 0).xyz;
  vec2 smoke = texelFetch(u_smokeTexture, coord, 0).xy;
  vec3 buoyancy = (u_densityScale * smoke.x
    - u_temperatureScale * (smoke.y - AMBIENT_TEMPERATURE)) * GRAVITY;
  o_velocity = velocity + u_deltaTime * buoyancy;
}
`;

  const ADVECT_VELOCITY_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec3 o_velocity;

uniform int u_cellNum;
uniform int u_cellTextureSize;
uniform ivec3 u_resolution;
uniform sampler2D u_velocityTexture;
uniform float u_deltaTime;
uniform float u_gridSpacing;

ivec2 convertCellIdToCoord(int cellId) {
  return ivec2(cellId % u_cellTextureSize, cellId / u_cellTextureSize);
}

int convertCoordToCellId(ivec2 coord) {
  return coord.x + coord.y * u_cellTextureSize;
}

int convertCellIndexToCellId(ivec3 cellIndex) {
  return  cellIndex.x + cellIndex.y * u_resolution.x + cellIndex.z * (u_resolution.x * u_resolution.y);
}

ivec3 convertCellIdToCellIndex(int cellId) {
  int z = cellId / (u_resolution.x * u_resolution.y);
  int y = (cellId % (u_resolution.x * u_resolution.y)) / u_resolution.x;
  int x = cellId % u_resolution.x;
  return ivec3(x, y, z);
}

vec3 convertCellIdToPosition(int cellId) {
  ivec3 cellIndex = convertCellIdToCellIndex(cellId);
  return (vec3(cellIndex) + 0.5) * u_gridSpacing;
}

ivec3 convertPositionToCellIndex(vec3 position) {
  return ivec3(position / u_gridSpacing - 0.5);
}

int convertPositionToCellId(vec3 position) {
  ivec3 cellIndex = convertPositionToCellIndex(position);
  return convertCellIndexToCellId(cellIndex);
}

ivec2 convertPositionToCoord(vec3 position) {
  int cellId = convertPositionToCellId(position);
  return convertCellIdToCoord(cellId);
}

ivec2 convertCellIndexToCoord(ivec3 cellIndex) {
  int cellId = convertCellIndexToCellId(cellIndex);
  return convertCellIdToCoord(cellId);
}

vec3 sampleVelocity(ivec3 cellIndex) {
  if (cellIndex.x < 0 || cellIndex.x >= u_resolution.x ||
      cellIndex.y < 0 || cellIndex.y >= u_resolution.y ||
      cellIndex.z < 0 || cellIndex.z >= u_resolution.z) {
    return vec3(0.0);
  }
  ivec2 coord = convertCellIndexToCoord(cellIndex);
  return texelFetch(u_velocityTexture, coord, 0).xyz;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int cellId = convertCoordToCellId(coord);
  if (cellId >= u_cellNum) {
    return;
  }
  vec3 position = convertCellIdToPosition(cellId);
  vec3 velocity = texelFetch(u_velocityTexture, coord, 0).xyz;

  vec3 prevPos = position - u_deltaTime * velocity;
  vec3 prevCellIndex = prevPos / u_gridSpacing - 0.5;
  ivec3 i = ivec3(prevCellIndex);
  vec3 f = fract(prevCellIndex);

  vec3 vel000 = sampleVelocity(i);
  vec3 vel100 = sampleVelocity(i + ivec3(1, 0, 0));
  vec3 vel010 = sampleVelocity(i + ivec3(0, 1, 0));
  vec3 vel110 = sampleVelocity(i + ivec3(1, 1, 0));
  vec3 vel001 = sampleVelocity(i + ivec3(0, 0, 1));
  vec3 vel101 = sampleVelocity(i + ivec3(1, 0, 1));
  vec3 vel011 = sampleVelocity(i + ivec3(0, 1, 1));
  vec3 vel111 = sampleVelocity(i + ivec3(1, 1, 1));

  o_velocity = mix(
    mix(mix(vel000, vel100, f.x), mix(vel010, vel110, f.x), f.y),
    mix(mix(vel001, vel101, f.x), mix(vel011, vel111, f.x), f.y),
    f.z
  );
}
`;

  const COMPUTE_PRESSURE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out float o_pressure;

uniform int u_cellNum;
uniform int u_cellTextureSize;
uniform ivec3 u_resolution;
uniform sampler2D u_velocityTexture;
uniform sampler2D u_pressureTexture;
uniform float u_deltaTime;
uniform float u_gridSpacing;
uniform float u_density;

ivec2 convertCellIdToCoord(int cellId) {
  return ivec2(cellId % u_cellTextureSize, cellId / u_cellTextureSize);
}

int convertCoordToCellId(ivec2 coord) {
  return coord.x + coord.y * u_cellTextureSize;
}

int convertCellIndexToCellId(ivec3 cellIndex) {
  return  cellIndex.x + cellIndex.y * u_resolution.x + cellIndex.z * (u_resolution.x * u_resolution.y);
}

ivec2 convertCellIndexToCoord(ivec3 cellIndex) {
  int cellId = convertCellIndexToCellId(cellIndex);
  return convertCellIdToCoord(cellId);
}

ivec3 convertCellIdToCellIndex(int cellId) {
  int z = cellId / (u_resolution.x * u_resolution.y);
  int y = (cellId % (u_resolution.x * u_resolution.y)) / u_resolution.x;
  int x = cellId % u_resolution.x;
  return ivec3(x, y, z);
}

void sampleVelocityAndPressure(ivec3 cellIndex, out vec3 velocity, out float pressure) {
  if (cellIndex.x < 0 || cellIndex.x >= u_resolution.x ||
      cellIndex.y < 0 || cellIndex.y >= u_resolution.y ||
      cellIndex.z < 0 || cellIndex.z >= u_resolution.z) {
    velocity = vec3(0.0);
    pressure = 0.0;
    return;
  }
  ivec2 coord = convertCellIndexToCoord(cellIndex);
  velocity = texelFetch(u_velocityTexture, coord, 0).xyz;
  pressure = texelFetch(u_pressureTexture, coord, 0).x;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int cellId = convertCoordToCellId(coord);
  if (cellId >= u_cellNum) {
    return;
  }

  ivec3 cellIndex = convertCellIdToCellIndex(cellId);
  vec3 vl, vr, vd, vu, vn, vf;
  float pl, pr, pd, pu, pn, pf;
  sampleVelocityAndPressure(cellIndex + ivec3(-1, 0, 0), vl, pl);
  sampleVelocityAndPressure(cellIndex + ivec3(1, 0, 0), vr, pr);
  sampleVelocityAndPressure(cellIndex + ivec3(0, -1, 0), vd, pd);
  sampleVelocityAndPressure(cellIndex + ivec3(0, 1, 0), vu, pu);
  sampleVelocityAndPressure(cellIndex + ivec3(0, 0, -1), vn, pn);
  sampleVelocityAndPressure(cellIndex + ivec3(0, 0, 1), vf, pf);

  o_pressure = (pl + pr + pd + pu + pn + pf
    - 0.5 * (vr.x - vl.x + vu.y - vd.y + vf.z - vn.z)
    * u_gridSpacing * u_density / u_deltaTime) / 6.0;
}
`;

  const ADD_PRESSURE_FORCE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec3 o_velocity;

uniform int u_cellNum;
uniform int u_cellTextureSize;
uniform ivec3 u_resolution;
uniform sampler2D u_velocityTexture;
uniform sampler2D u_pressureTexture;
uniform float u_deltaTime;
uniform float u_gridSpacing;
uniform float u_density;

ivec2 convertCellIdToCoord(int cellId) {
  return ivec2(cellId % u_cellTextureSize, cellId / u_cellTextureSize);
}

int convertCoordToCellId(ivec2 coord) {
  return coord.x + coord.y * u_cellTextureSize;
}

int convertCellIndexToCellId(ivec3 cellIndex) {
  return  cellIndex.x + cellIndex.y * u_resolution.x + cellIndex.z * (u_resolution.x * u_resolution.y);
}

ivec3 convertCellIdToCellIndex(int cellId) {
  int z = cellId / (u_resolution.x * u_resolution.y);
  int y = (cellId % (u_resolution.x * u_resolution.y)) / u_resolution.x;
  int x = cellId % u_resolution.x;
  return ivec3(x, y, z);
}

ivec2 convertCellIndexToCoord(ivec3 cellIndex) {
  int cellId = convertCellIndexToCellId(cellIndex);
  return convertCellIdToCoord(cellId);
}

float samplePressure(ivec3 cellIndex) {
  if (cellIndex.x < 0) {
    cellIndex.x = 0;
  }
  if (cellIndex.x >= u_resolution.x) {
    cellIndex.x = u_resolution.x - 1;
  }
  if (cellIndex.y < 0) {
    cellIndex.y = 0;
  }
  if (cellIndex.y >= u_resolution.y) {
    cellIndex.y = u_resolution.y - 1;
  }
  if (cellIndex.z < 0) {
    cellIndex.z = 0;
  }
  if (cellIndex.z >= u_resolution.z) {
    cellIndex.z = u_resolution.z - 1;
  }
  ivec2 coord = convertCellIndexToCoord(cellIndex);
  return texelFetch(u_pressureTexture, coord, 0).x;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int cellId = convertCoordToCellId(coord);
  if (cellId >= u_cellNum) {
    return;
  }

  vec3 velocity = texelFetch(u_velocityTexture, coord, 0).xyz;

  ivec3 cellIndex = convertCellIdToCellIndex(cellId);
  float pl = samplePressure(cellIndex + ivec3(-1, 0, 0));
  float pr = samplePressure(cellIndex + ivec3(1, 0, 0));
  float pd = samplePressure(cellIndex + ivec3(0, -1, 0));
  float pu = samplePressure(cellIndex + ivec3(0, 1, 0));
  float pn = samplePressure(cellIndex + ivec3(0, 0, -1));
  float pf = samplePressure(cellIndex + ivec3(0, 0, 1));

  o_velocity = velocity - 0.5 * u_deltaTime * vec3(pr - pl, pu - pd, pf - pn) / (u_gridSpacing * u_density);
}
`;

  const DECAY_VELOCITY_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec3 o_velocity;

uniform sampler2D u_velocityTexture;
uniform float u_deltaTime;
uniform float u_velocityDecay;

void main(void) {
  vec3 velocity = texelFetch(u_velocityTexture, ivec2(gl_FragCoord.xy), 0).xyz;
  velocity *= exp(-u_velocityDecay * u_deltaTime);
  o_velocity = velocity;
}
`;

  const ADVECT_SMOKE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec2 o_smoke;

uniform int u_cellNum;
uniform int u_cellTextureSize;
uniform ivec3 u_resolution;
uniform sampler2D u_velocityTexture;
uniform sampler2D u_smokeTexture;
uniform float u_deltaTime;
uniform float u_gridSpacing;

#define AMBIENT_TEMPERATURE 273.0

ivec2 convertCellIdToCoord(int cellId) {
  return ivec2(cellId % u_cellTextureSize, cellId / u_cellTextureSize);
}

int convertCoordToCellId(ivec2 coord) {
  return coord.x + coord.y * u_cellTextureSize;
}

int convertCellIndexToCellId(ivec3 cellIndex) {
  return  cellIndex.x + cellIndex.y * u_resolution.x + cellIndex.z * (u_resolution.x * u_resolution.y);
}

ivec3 convertCellIdToCellIndex(int cellId) {
  int z = cellId / (u_resolution.x * u_resolution.y);
  int y = (cellId % (u_resolution.x * u_resolution.y)) / u_resolution.x;
  int x = cellId % u_resolution.x;
  return ivec3(x, y, z);
}

vec3 convertCellIdToPosition(int cellId) {
  ivec3 cellIndex = convertCellIdToCellIndex(cellId);
  return (vec3(cellIndex) + 0.5) * u_gridSpacing;
}

ivec3 convertPositionToCellIndex(vec3 position) {
  return ivec3(position / u_gridSpacing - 0.5);
}

int convertPositionToCellId(vec3 position) {
  ivec3 cellIndex = convertPositionToCellIndex(position);
  return convertCellIndexToCellId(cellIndex);
}

ivec2 convertPositionToCoord(vec3 position) {
  int cellId = convertPositionToCellId(position);
  return convertCellIdToCoord(cellId);
}

ivec2 convertCellIndexToCoord(ivec3 cellIndex) {
  int cellId = convertCellIndexToCellId(cellIndex);
  return convertCellIdToCoord(cellId);
}

vec2 sampleSmoke(ivec3 cellIndex) {
  if (cellIndex.x < 0 || cellIndex.x >= u_resolution.x ||
      cellIndex.y < 0 || cellIndex.y >= u_resolution.y ||
      cellIndex.z < 0 || cellIndex.z >= u_resolution.z) {
    return vec2(0.0, AMBIENT_TEMPERATURE);
  }
  ivec2 coord = convertCellIndexToCoord(cellIndex);
  return texelFetch(u_smokeTexture, coord, 0).xy;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int cellId = convertCoordToCellId(coord);
  if (cellId >= u_cellNum) {
    return;
  }
  vec3 position = convertCellIdToPosition(cellId);
  vec3 velocity = texelFetch(u_velocityTexture, coord, 0).xyz;

  vec3 prevPos = position - u_deltaTime * velocity;
  vec3 prevCellIndex = prevPos / u_gridSpacing - 0.5;
  ivec3 i = ivec3(prevCellIndex);
  vec3 f = fract(prevCellIndex);

  vec2 smoke000 = sampleSmoke(i);
  vec2 smoke100 = sampleSmoke(i + ivec3(1, 0, 0));
  vec2 smoke010 = sampleSmoke(i + ivec3(0, 1, 0));
  vec2 smoke110 = sampleSmoke(i + ivec3(1, 1, 0));
  vec2 smoke001 = sampleSmoke(i + ivec3(0, 0, 1));
  vec2 smoke101 = sampleSmoke(i + ivec3(1, 0, 1));
  vec2 smoke011 = sampleSmoke(i + ivec3(0, 1, 1));
  vec2 smoke111 = sampleSmoke(i + ivec3(1, 1, 1));

  o_smoke = mix(
    mix(mix(smoke000, smoke100, f.x), mix(smoke010, smoke110, f.x), f.y),
    mix(mix(smoke001, smoke101, f.x), mix(smoke011, smoke111, f.x), f.y),
    f.z
  );
}
`;

  const ADD_SMOKE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec2 o_smoke;

uniform int u_cellNum;
uniform int u_cellTextureSize;
uniform ivec3 u_resolution;
uniform vec3 u_simulationSpace;
uniform sampler2D u_smokeTexture;
uniform float u_deltaTime;
uniform float u_gridSpacing;
uniform bool u_addHeat;
uniform vec2 u_mousePosition;
uniform float u_heatSourceRadius;
uniform float u_heatSourceIntensity;
uniform float u_densityDecay;
uniform float u_temperatureDecay;

#define AMBIENT_TEMPERATURE 273.0

ivec2 convertCellIdToCoord(int cellId) {
  return ivec2(cellId % u_cellTextureSize, cellId / u_cellTextureSize);
}

int convertCoordToCellId(ivec2 coord) {
  return coord.x + coord.y * u_cellTextureSize;
}

ivec3 convertCellIdToCellIndex(int cellId) {
  int z = cellId / (u_resolution.x * u_resolution.y);
  int y = (cellId % (u_resolution.x * u_resolution.y)) / u_resolution.x;
  int x = cellId % u_resolution.x;
  return ivec3(x, y, z);
}

vec3 convertCellIdToPosition(int cellId) {
  ivec3 cellIndex = convertCellIdToCellIndex(cellId);
  return (vec3(cellIndex) + 0.5) * u_gridSpacing;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int cellId = convertCoordToCellId(coord);
  if (cellId >= u_cellNum) {
    return;
  }

  vec2 smoke = texelFetch(u_smokeTexture, convertCellIdToCoord(cellId), 0).xy;
  float density = smoke.x;
  float temperature = smoke.y;

  float nextTemperature = temperature;
  vec3 position = convertCellIdToPosition(cellId);
  if (u_addHeat) {
    vec3 heatCenter = vec3(u_mousePosition.x * u_simulationSpace.x, u_simulationSpace.y * 0.25, u_mousePosition.y * u_simulationSpace.z);
    nextTemperature += smoothstep(u_heatSourceRadius, 0.0, length(position - heatCenter))
      * u_deltaTime * u_heatSourceIntensity;
  }
  nextTemperature += (1.0 - exp(-u_temperatureDecay * u_deltaTime)) * (AMBIENT_TEMPERATURE - nextTemperature);

  float nextDensity = density + u_deltaTime * max(0.0, (nextTemperature - (AMBIENT_TEMPERATURE + 100.0))) * 0.01;
  nextDensity *= exp(-u_densityDecay * u_deltaTime);

  o_smoke = vec2(nextDensity, nextTemperature);
}
`;

  const RAYMARCH_VERTEX_SHADER_SOURCE =
`#version 300 es

const vec3[8] CUBE_POSITIONS = vec3[](
  vec3(-1.0, -1.0,  1.0),
  vec3( 1.0, -1.0,  1.0),
  vec3( 1.0, -1.0, -1.0),
  vec3(-1.0, -1.0, -1.0),
  vec3(-1.0,  1.0,  1.0),
  vec3( 1.0,  1.0,  1.0),
  vec3( 1.0,  1.0, -1.0),
  vec3(-1.0,  1.0, -1.0)
);

const vec3[6] CUBE_NORMALS = vec3[](
  vec3(0.0, 0.0, 1.0),
  vec3(1.0, 0.0, 0.0),
  vec3(0.0, 0.0, -1.0),
  vec3(-1.0, 0.0, 0.0),
  vec3(0.0, 1.0, 0.0),
  vec3(0.0, -1.0, 0.0)
);

const int[36] CUBE_INDICES = int[](
  0, 5, 4, 0, 1, 5,
  1, 6, 5, 1, 2, 6,
  2, 7, 6, 2, 3, 7,
  3, 4, 7, 3, 0, 4,
  4, 6, 7, 4, 5, 6,
  3, 1, 0, 3, 2, 1
);

out vec3 v_position;
out vec3 v_normal;

uniform mat4 u_mvpMatrix;
uniform mat4 u_modelMatrix;
uniform vec3 u_scale;

void main(void) {
  vec3 position = u_scale * CUBE_POSITIONS[CUBE_INDICES[gl_VertexID]];
  vec3 normal = CUBE_NORMALS[gl_VertexID / 6];
  v_position = (u_modelMatrix * vec4(position, 1.0)).xyz;
  v_normal = (u_modelMatrix * vec4(normal, 0.0)).xyz;
  gl_Position = u_mvpMatrix * vec4(position, 1.0);
}
`;

  const RENDER_VELOCITY_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec3 v_position;
in vec3 v_normal;

out vec4 o_color;

uniform mat4 u_mvpMatrix;
uniform mat4 u_modelMatrix;
uniform mat4 u_invModelMatrix;
uniform vec3 u_scale;
uniform vec3 u_cameraPosition;
uniform int u_cellTextureSize;
uniform ivec3 u_resolution;
uniform vec3 u_simulationSpace;
uniform sampler2D u_velocityTexture;
uniform float u_gridSpacing;

struct Ray {
  vec3 origin;
  vec3 dir;
};

Ray convertRayFromWorldToObject(Ray ray) {
  vec3 origin = (u_invModelMatrix * vec4(ray.origin, 1.0)).xyz;
  vec3 dir = normalize((u_invModelMatrix * vec4(ray.dir, 0.0)).xyz);
  return Ray(origin, dir);
}

void getRange(Ray ray, inout float tmin, inout float tmax) {
  for (int i = 0; i < 3; i++) {
    float t1 = (u_scale[i] - ray.origin[i]) / ray.dir[i];
    float t2 = (-u_scale[i] - ray.origin[i]) / ray.dir[i];
    tmin = max(tmin, min(t1, t2));
    tmax = min(tmax, max(t1, t2));
  }
}

int convertCellIndexToCellId(ivec3 cellIndex) {
  return  cellIndex.x + cellIndex.y * u_resolution.x + cellIndex.z * (u_resolution.x * u_resolution.y);
}

ivec2 convertCellIdToCoord(int cellId) {
  return ivec2(cellId % u_cellTextureSize, cellId / u_cellTextureSize);
}

ivec2 convertCellIndexToCoord(ivec3 cellIndex) {
  int cellId = convertCellIndexToCellId(cellIndex);
  return convertCellIdToCoord(cellId);
}

ivec3 convertPositionToCellIndex(vec3 position) {
  return ivec3(position / u_gridSpacing - 0.5);
}

vec3 convertToSimulationSpace(vec3 p) {
  p /= u_scale;
  p *= u_simulationSpace;
  p = (p + u_simulationSpace) * 0.5;
  return p;
}

vec3 sampleVelocity(ivec3 cellIndex) {
  if (cellIndex.x < 0 || cellIndex.x >= u_resolution.x ||
      cellIndex.y < 0 || cellIndex.y >= u_resolution.y ||
      cellIndex.z < 0 || cellIndex.z >= u_resolution.z) {
    return vec3(0.0);
  }
  ivec2 coord = convertCellIndexToCoord(cellIndex);
  return texelFetch(u_velocityTexture, coord, 0).xyz;
}

vec3 sampleVelocity(vec3 p) {
  vec3 cellIndex = convertToSimulationSpace(p) / u_gridSpacing;

  ivec3 i = ivec3(cellIndex) - (1 - int(step(0.5, cellIndex)));
  vec3 f = smoothstep(0.0, 1.0, fract(cellIndex + 0.5));

  vec3 vel000 = sampleVelocity(i);
  vec3 vel100 = sampleVelocity(i + ivec3(1, 0, 0));
  vec3 vel010 = sampleVelocity(i + ivec3(0, 1, 0));
  vec3 vel110 = sampleVelocity(i + ivec3(1, 1, 0));
  vec3 vel001 = sampleVelocity(i + ivec3(0, 0, 1));
  vec3 vel101 = sampleVelocity(i + ivec3(1, 0, 1));
  vec3 vel011 = sampleVelocity(i + ivec3(0, 1, 1));
  vec3 vel111 = sampleVelocity(i + ivec3(1, 1, 1));

  return mix(
    mix(mix(vel000, vel100, f.x), mix(vel010, vel110, f.x), f.y),
    mix(mix(vel001, vel101, f.x), mix(vel011, vel111, f.x), f.y),
    f.z
  );
}

#define RAYMARCH_ITERATIONS 48

vec4 raymarch(vec3 ro, vec3 rd, float tmin, float tmax) {
  float raymarchSize = (2.0 * length(u_scale)) / float(RAYMARCH_ITERATIONS);
  vec3 p = ro + (tmin + (raymarchSize - mod(tmin, raymarchSize))) * rd;
  vec3 color = vec3(0.0);
  float transmittance = 1.0;
  for (int ri = 0; ri < RAYMARCH_ITERATIONS; ri++) {
    vec3 velocity = sampleVelocity(p);
    float density = clamp(length(velocity) * 20.0, 0.0, 1.0);
    color += (clamp(velocity * 20.0, -1.0, 1.0) * 0.5 + 0.5) * transmittance * density;
    transmittance *= 1.0 - density;
    if (transmittance < 0.001) {
      break;
    }
    p += raymarchSize * rd;
  }
  return vec4(color, 1.0 - transmittance);
}

void main(void) {
  vec3 dir = normalize(v_position - u_cameraPosition);
  Ray ray = convertRayFromWorldToObject(Ray(u_cameraPosition, dir));
  float tmin = 0.0;
  float tmax = 1e16;
  getRange(ray, tmin, tmax);
  vec4 c = raymarch(ray.origin, ray.dir, tmin, tmax);
  if (c.w > 0.0) {
    o_color = vec4(c.rgb, c.a);
  } else {
    discard;
  }
}
`;

const RENDER_DENSITY_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec3 v_position;
in vec3 v_normal;

out vec4 o_color;

uniform mat4 u_mvpMatrix;
uniform mat4 u_modelMatrix;
uniform mat4 u_invModelMatrix;
uniform vec3 u_scale;
uniform vec3 u_cameraPosition;
uniform int u_cellTextureSize;
uniform ivec3 u_resolution;
uniform vec3 u_simulationSpace;
uniform sampler2D u_smokeTexture;
uniform float u_gridSpacing;

struct Ray {
  vec3 origin;
  vec3 dir;
};

Ray convertRayFromWorldToObject(Ray ray) {
  vec3 origin = (u_invModelMatrix * vec4(ray.origin, 1.0)).xyz;
  vec3 dir = normalize((u_invModelMatrix * vec4(ray.dir, 0.0)).xyz);
  return Ray(origin, dir);
}

void getRange(Ray ray, inout float tmin, inout float tmax) {
  for (int i = 0; i < 3; i++) {
    float t1 = (u_scale[i] - ray.origin[i]) / ray.dir[i];
    float t2 = (-u_scale[i] - ray.origin[i]) / ray.dir[i];
    tmin = max(tmin, min(t1, t2));
    tmax = min(tmax, max(t1, t2));
  }
}

int convertCellIndexToCellId(ivec3 cellIndex) {
  return  cellIndex.x + cellIndex.y * u_resolution.x + cellIndex.z * (u_resolution.x * u_resolution.y);
}

ivec2 convertCellIdToCoord(int cellId) {
  return ivec2(cellId % u_cellTextureSize, cellId / u_cellTextureSize);
}

ivec2 convertCellIndexToCoord(ivec3 cellIndex) {
  int cellId = convertCellIndexToCellId(cellIndex);
  return convertCellIdToCoord(cellId);
}

ivec3 convertPositionToCellIndex(vec3 position) {
  return ivec3(position / u_gridSpacing - 0.5);
}

vec3 convertToSimulationSpace(vec3 p) {
  p /= u_scale;
  p *= u_simulationSpace;
  p = (p + u_simulationSpace) * 0.5;
  return p;
}

float sampleDensity(ivec3 cellIndex) {
  if (cellIndex.x < 0 || cellIndex.x >= u_resolution.x ||
      cellIndex.y < 0 || cellIndex.y >= u_resolution.y ||
      cellIndex.z < 0 || cellIndex.z >= u_resolution.z) {
    return 0.0;
  }
  ivec2 coord = convertCellIndexToCoord(cellIndex);
  return texelFetch(u_smokeTexture, coord, 0).x;
}

float sampleDensity(vec3 p) {
  vec3 cellIndex = convertToSimulationSpace(p) / u_gridSpacing;

  ivec3 i = ivec3(cellIndex) - (1 - int(step(0.5, cellIndex)));
  vec3 f = smoothstep(0.0, 1.0, fract(cellIndex + 0.5));

  float dens000 = sampleDensity(i);
  float dens100 = sampleDensity(i + ivec3(1, 0, 0));
  float dens010 = sampleDensity(i + ivec3(0, 1, 0));
  float dens110 = sampleDensity(i + ivec3(1, 1, 0));
  float dens001 = sampleDensity(i + ivec3(0, 0, 1));
  float dens101 = sampleDensity(i + ivec3(1, 0, 1));
  float dens011 = sampleDensity(i + ivec3(0, 1, 1));
  float dens111 = sampleDensity(i + ivec3(1, 1, 1));

  return mix(
    mix(mix(dens000, dens100, f.x), mix(dens010, dens110, f.x), f.y),
    mix(mix(dens001, dens101, f.x), mix(dens011, dens111, f.x), f.y),
    f.z
  );
}

#define RAYMARCH_ITERATIONS 128
#define SHADOW_ITERATIONS 4
#define SHADOW_LENGTH u_simulationSpace.y * 1.0

#define LIGHT_DIR normalize(vec3(0.0, 1.0, 0.0))
#define LIGHT_COLOR vec3(1.0, 1.0, 0.95)

vec4 raymarch(vec3 ro, vec3 rd, float tmin, float tmax) {
  float raymarchSize = (length(2.0 * u_scale)) / float(RAYMARCH_ITERATIONS);
  float shadowSize = SHADOW_LENGTH / float(SHADOW_ITERATIONS);
  vec3 rayStep = rd * raymarchSize;
  vec3 shadowStep = LIGHT_DIR * shadowSize;
  vec3 p = ro + (tmin + (raymarchSize - mod(tmin, raymarchSize))) * rd;
  vec3 color = vec3(0.0);
  float transmittance = 1.0;
  for (int ri = 0; ri < RAYMARCH_ITERATIONS; ri++) {
    float density = clamp(sampleDensity(p), 0.0, 1.0);
    if (density > 0.001) {
      vec3 shadowPosition = p;
      float shadowDensity = 0.0;
      for(int si = 0; si < SHADOW_ITERATIONS; si++) {
        shadowPosition += shadowStep;
        shadowDensity = sampleDensity(shadowPosition);
      }
      float atten = exp(-shadowDensity * 1.0);
      vec3 attenLight = LIGHT_COLOR * atten;
      color += vec3(0.95, 0.98, 1.0) * attenLight * transmittance * density; 

      transmittance *= 1.0 - density;
    }
    if (transmittance < 0.001) {
      break;
    }
    p += rayStep;
  }
  return vec4(color, 1.0 - transmittance);
}

void main(void) {
  vec3 dir = normalize(v_position - u_cameraPosition);
  Ray ray = convertRayFromWorldToObject(Ray(u_cameraPosition, dir));
  float tmin = 0.0;
  float tmax = 1e16;
  getRange(ray, tmin, tmax);
  vec4 c = raymarch(ray.origin, ray.dir, tmin, tmax);
  if (c.w > 0.0) {
    o_color = vec4(pow(c.rgb, vec3(1.0 / 2.2)), c.w);
  } else {
    discard;
  }
}
`;

  const RENDER_TEMPERATURE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec3 v_position;
in vec3 v_normal;

out vec4 o_color;

uniform mat4 u_mvpMatrix;
uniform mat4 u_modelMatrix;
uniform mat4 u_invModelMatrix;
uniform vec3 u_scale;
uniform vec3 u_cameraPosition;
uniform int u_cellTextureSize;
uniform ivec3 u_resolution;
uniform vec3 u_simulationSpace;
uniform sampler2D u_smokeTexture;
uniform float u_gridSpacing;

#define AMBIENT_TEMPERATURE 273.0

struct Ray {
  vec3 origin;
  vec3 dir;
};

Ray convertRayFromWorldToObject(Ray ray) {
  vec3 origin = (u_invModelMatrix * vec4(ray.origin, 1.0)).xyz;
  vec3 dir = normalize((u_invModelMatrix * vec4(ray.dir, 0.0)).xyz);
  return Ray(origin, dir);
}

void getRange(Ray ray, inout float tmin, inout float tmax) {
  for (int i = 0; i < 3; i++) {
    float t1 = (u_scale[i] - ray.origin[i]) / ray.dir[i];
    float t2 = (-u_scale[i] - ray.origin[i]) / ray.dir[i];
    tmin = max(tmin, min(t1, t2));
    tmax = min(tmax, max(t1, t2));
  }
}

int convertCellIndexToCellId(ivec3 cellIndex) {
  return  cellIndex.x + cellIndex.y * u_resolution.x + cellIndex.z * (u_resolution.x * u_resolution.y);
}

ivec2 convertCellIdToCoord(int cellId) {
  return ivec2(cellId % u_cellTextureSize, cellId / u_cellTextureSize);
}

ivec2 convertCellIndexToCoord(ivec3 cellIndex) {
  int cellId = convertCellIndexToCellId(cellIndex);
  return convertCellIdToCoord(cellId);
}

ivec3 convertPositionToCellIndex(vec3 position) {
  return ivec3(position / u_gridSpacing - 0.5);
}

vec3 convertToSimulationSpace(vec3 p) {
  p /= u_scale;
  p *= u_simulationSpace;
  p = (p + u_simulationSpace) * 0.5;
  return p;
}

float sampleTemperature(ivec3 cellIndex) {
  if (cellIndex.x < 0 || cellIndex.x >= u_resolution.x ||
      cellIndex.y < 0 || cellIndex.y >= u_resolution.y ||
      cellIndex.z < 0 || cellIndex.z >= u_resolution.z) {
    return AMBIENT_TEMPERATURE;
  }
  ivec2 coord = convertCellIndexToCoord(cellIndex);
  return texelFetch(u_smokeTexture, coord, 0).y;
}

float sampleTemperature(vec3 p) {
  vec3 cellIndex = convertToSimulationSpace(p) / u_gridSpacing;

  ivec3 i = ivec3(cellIndex) - (1 - int(step(0.5, cellIndex)));
  vec3 f = smoothstep(0.0, 1.0, fract(cellIndex + 0.5));

  float temp000 = sampleTemperature(i);
  float temp100 = sampleTemperature(i + ivec3(1, 0, 0));
  float temp010 = sampleTemperature(i + ivec3(0, 1, 0));
  float temp110 = sampleTemperature(i + ivec3(1, 1, 0));
  float temp001 = sampleTemperature(i + ivec3(0, 0, 1));
  float temp101 = sampleTemperature(i + ivec3(1, 0, 1));
  float temp011 = sampleTemperature(i + ivec3(0, 1, 1));
  float temp111 = sampleTemperature(i + ivec3(1, 1, 1));

  return mix(
    mix(mix(temp000, temp100, f.x), mix(temp010, temp110, f.x), f.y),
    mix(mix(temp001, temp101, f.x), mix(temp011, temp111, f.x), f.y),
    f.z
  );
}

#define RAYMARCH_ITERATIONS 128

const vec4[6] TEMPERATURE_COLOR = vec4[](
  vec4(0.0, 0.0, 0.0, 0.0),
  vec4(0.0, 0.0, 0.0, AMBIENT_TEMPERATURE),
  vec4(1.0, 0.0, 0.0, AMBIENT_TEMPERATURE + 100.0),
  vec4(1.0, 0.5, 0.0, AMBIENT_TEMPERATURE + 200.0),
  vec4(1.0, 1.0, 1.0, AMBIENT_TEMPERATURE + 300.0),
  vec4(0.5, 0.5, 1.0, AMBIENT_TEMPERATURE + 400.0)
);

vec3 getTemperatureColor(float temperature) {
  vec3 color = TEMPERATURE_COLOR[5].xyz;
  for (int i = 0; i < 5; i++) {
    if (temperature < TEMPERATURE_COLOR[i + 1].w) {
      color = mix(TEMPERATURE_COLOR[i].xyz, TEMPERATURE_COLOR[i + 1].xyz,
        1.0 - (TEMPERATURE_COLOR[i + 1].w - temperature) / (TEMPERATURE_COLOR[i + 1]. w - TEMPERATURE_COLOR[i].w));
      break;
    }
  }
  return color;
}

vec4 raymarch(vec3 ro, vec3 rd, float tmin, float tmax) {
  float raymarchSize = (2.0 * length(u_scale)) / float(RAYMARCH_ITERATIONS);
  vec3 p = ro + (tmin + (raymarchSize - mod(tmin, raymarchSize))) * rd;
  vec3 color = vec3(0.0);
  float transmittance = 1.0;
  for (int ri = 0; ri < RAYMARCH_ITERATIONS; ri++) {
    float temperature = sampleTemperature(p);
    vec3 c = getTemperatureColor(temperature);
    float density = clamp((temperature - AMBIENT_TEMPERATURE) / 500.0, 0.0, 1.0);
    color += c * transmittance * density;
    transmittance *= 1.0 - density;
    if (transmittance < 0.001) {
      break;
    }
    p += raymarchSize * rd;
  }
  return vec4(color, 1.0 - transmittance);
}

void main(void) {
  vec3 dir = normalize(v_position - u_cameraPosition);
  Ray ray = convertRayFromWorldToObject(Ray(u_cameraPosition, dir));
  float tmin = 0.0;
  float tmax = 1e16;
  getRange(ray, tmin, tmax);
  vec4 c = raymarch(ray.origin, ray.dir, tmin, tmax);
  if (c.w > 0.0) {
    o_color = vec4(c.rgb, c.w);
  } else {
    discard;
  }
}
`;

  let mousePosition = new Vector2(0.0, 0.0);
  let mousePressing = false;
  window.addEventListener('mousemove', event => {
    mousePosition = new Vector2(event.clientX, window.innerHeight - event.clientY);
  });
  window.addEventListener('mousedown', _ => {
    mousePressing = true;
  });
  window.addEventListener('mouseup', _ => {
    mousePressing = false;
  });

  const stats = new Stats();
  document.body.appendChild(stats.dom);

  const parameters = {
    'air density': 2.354, 
    'density force': 0.02,
    'temperature force': 0.0001,
    'heat radius': 0.05,
    'heat intensity': 1000.0,
    'velocity decay': 0.1,
    'density decay': 0.3,
    'temperature decay': 0.5,
    'time step': 0.005,
    'time scale': 0.5,
    'render': 'density',
    'reset': _ => reset()
  };

  const gui = new dat.GUI();
  gui.add(parameters, 'density force', 0.0, 0.1).step(0.0001);
  gui.add(parameters, 'temperature force', 0.0, 0.0003).step(0.00001);
  gui.add(parameters, 'heat radius', 0.0, 0.1).step(0.001);
  gui.add(parameters, 'heat intensity', 0.0, 2000.0).step(1.0);
  gui.add(parameters, 'velocity decay', 0.0, 2.0).step(0.1);
  gui.add(parameters, 'density decay', 0.0, 2.0).step(0.1);
  gui.add(parameters, 'temperature decay', 0.0, 2.0).step(0.1);
  gui.add(parameters, 'time step', 0.0001, 0.01).step(0.0001);
  gui.add(parameters, 'time scale', 0.5, 2.0).step(0.001);
  gui.add(parameters, 'render', ['density', 'temperature', 'velocity']);
  gui.add(parameters, 'reset');

  const SIMULATION_RESOLUTION = new Vector3(50, 50, 50);
  const GRID_SPACING = 0.005;
  const SIMULATION_SPACE = Vector3.mul(SIMULATION_RESOLUTION, GRID_SPACING);
  const CELL_NUM = SIMULATION_RESOLUTION.x * SIMULATION_RESOLUTION.y * SIMULATION_RESOLUTION.z;

  let cellTextureSize;
  for (let i = 0; ; i++) {
    cellTextureSize = 2 ** i;
    if (cellTextureSize * cellTextureSize >= CELL_NUM) {
      break;
    }
  }

  const canvas = document.getElementById('canvas');
  const resizeCanvas = function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  };
  window.addEventListener('resize', _ => {
    resizeCanvas();
  });
  resizeCanvas();

  const gl = canvas.getContext('webgl2');
  gl.getExtension('EXT_color_buffer_float');
  gl.clearColor(0.7, 0.7, 0.7, 1.0);

  const initializeVelocityProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, INITIALIZE_VELOCITY_FRAGMENT_SHADER_SOURCE);
  const initializeSmokeProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, INITIALIZE_SMOKE_FRAGMENT_SHADER_SOURCE);
  const addBuoyancyForceProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, ADD_BUOYANCY_FORCE_FRAGMENT_SHADER_SOURCE);
  const advectVelocityProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, ADVECT_VELOCITY_FRAGMENT_SHADER_SOURCE);
  const computePressureProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, COMPUTE_PRESSURE_FRAGMENT_SHADER_SOURCE);
  const addPressureForceProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, ADD_PRESSURE_FORCE_FRAGMENT_SHADER_SOURCE);
  const decayVelocityProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, DECAY_VELOCITY_FRAGMENT_SHADER_SOURCE);
  const advectSmokeProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, ADVECT_SMOKE_FRAGMENT_SHADER_SOURCE);
  const addSmokeProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, ADD_SMOKE_FRAGMENT_SHADER_SOURCE);
  const renderVelocityProgram = createProgramFromSource(gl, RAYMARCH_VERTEX_SHADER_SOURCE, RENDER_VELOCITY_FRAGMENT_SHADER_SOURCE);
  const renderDensityProgram = createProgramFromSource(gl, RAYMARCH_VERTEX_SHADER_SOURCE, RENDER_DENSITY_FRAGMENT_SHADER_SOURCE);
  const renderTemperatureProgram = createProgramFromSource(gl, RAYMARCH_VERTEX_SHADER_SOURCE, RENDER_TEMPERATURE_FRAGMENT_SHADER_SOURCE);

  const initializeSmokeUniforms = getUniformLocations(gl, initializeSmokeProgram, ['u_cellNum', 'u_cellTextureSize', 'u_resolution', 'u_gridSpacing', 'u_simulationSpace']);
  const addBuoyancyForceUniforms = getUniformLocations(gl, addBuoyancyForceProgram, ['u_cellNum', 'u_cellTextureSize', 'u_velocityTexture', 'u_smokeTexture', 'u_deltaTime', 'u_densityScale', 'u_temperatureScale']);
  const advectVelocityUniforms = getUniformLocations(gl, advectVelocityProgram,
    ['u_cellNum', 'u_cellTextureSize', 'u_resolution', 'u_velocityTexture', 'u_deltaTime', 'u_gridSpacing']);
  const computePressureUniforms = getUniformLocations(gl, computePressureProgram, ['u_cellNum', 'u_cellTextureSize', 'u_resolution', 'u_velocityTexture', 'u_pressureTexture', 'u_deltaTime', 'u_gridSpacing', 'u_density']);
  const addPressureForceUniforms = getUniformLocations(gl, addPressureForceProgram, ['u_cellNum', 'u_cellTextureSize', 'u_resolution', 'u_velocityTexture', 'u_pressureTexture', 'u_deltaTime', 'u_gridSpacing', 'u_density']);
  const decayVelocityUniforms = getUniformLocations(gl, decayVelocityProgram, ['u_velocityTexture', 'u_deltaTime', 'u_velocityDecay']);
  const advectSmokeUniforms = getUniformLocations(gl, advectSmokeProgram,
    ['u_cellNum', 'u_cellTextureSize', 'u_resolution', 'u_velocityTexture', 'u_smokeTexture', 'u_deltaTime', 'u_gridSpacing']);
  const addSmokeUniforms = getUniformLocations(gl, addSmokeProgram,
    ['u_cellNum', 'u_cellTextureSize', 'u_resolution', 'u_simulationSpace', 'u_smokeTexture', 'u_deltaTime', 'u_gridSpacing', 'u_addHeat', 'u_mousePosition', 'u_heatSourceRadius', 'u_heatSourceIntensity', 'u_densityDecay', 'u_temperatureDecay']);
  const renderVelocityUniforms = getUniformLocations(gl, renderVelocityProgram,
    ['u_mvpMatrix', 'u_modelMatrix', 'u_invModelMatrix', 'u_scale', 'u_cameraPosition',
     'u_cellTextureSize', 'u_resolution', 'u_simulationSpace', 'u_velocityTexture', 'u_gridSpacing']);
  const renderDensityUniforms = getUniformLocations(gl, renderDensityProgram,
    ['u_mvpMatrix', 'u_modelMatrix', 'u_invModelMatrix', 'u_scale', 'u_cameraPosition',
     'u_cellTextureSize', 'u_resolution', 'u_simulationSpace', 'u_smokeTexture', 'u_gridSpacing']);
  const renderTemperatureUniforms = getUniformLocations(gl, renderTemperatureProgram,
    ['u_mvpMatrix', 'u_modelMatrix', 'u_invModelMatrix', 'u_scale', 'u_cameraPosition',
     'u_cellTextureSize', 'u_resolution', 'u_simulationSpace', 'u_smokeTexture', 'u_gridSpacing']);

  let requestId = null;
  const reset = function() {
    if (requestId !== null) {
      cancelAnimationFrame(requestId);
      requestId = null;
    }

    let velocityFbObjR = createVelocityFramebuffer(gl, cellTextureSize, cellTextureSize);
    let velocityFbObjW = createVelocityFramebuffer(gl, cellTextureSize, cellTextureSize);
    const swapVelocityFbObj = function() {
      const tmp = velocityFbObjR;
      velocityFbObjR = velocityFbObjW;
      velocityFbObjW = tmp;
    };

    let pressureFbObjR = createPressureFramebuffer(gl, cellTextureSize, cellTextureSize);
    let pressureFbObjW = createPressureFramebuffer(gl, cellTextureSize, cellTextureSize);
    const swapPressureFbObj = function() {
      const tmp = pressureFbObjR;
      pressureFbObjR = pressureFbObjW;
      pressureFbObjW = tmp;
    };


    let smokeFbObjR = createSmokeFramebuffer(gl, cellTextureSize, cellTextureSize);
    let smokeFbObjW = createSmokeFramebuffer(gl, cellTextureSize, cellTextureSize);
    const swapSmokeFbObj = function() {
      const tmp = smokeFbObjR;
      smokeFbObjR = smokeFbObjW;
      smokeFbObjW = tmp;
    };

    const initializeVelocity = function() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, velocityFbObjW.framebuffer);
      gl.useProgram(initializeVelocityProgram);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapVelocityFbObj();
    };

    const initializeSmoke = function() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, smokeFbObjW.framebuffer);
      gl.useProgram(initializeSmokeProgram);
      gl.uniform1i(initializeSmokeUniforms['u_cellNum'], CELL_NUM);
      gl.uniform1i(initializeSmokeUniforms['u_cellTextureSize'], cellTextureSize);
      gl.uniform3iv(initializeSmokeUniforms['u_resolution'], SIMULATION_RESOLUTION.toArray());
      gl.uniform1f(initializeSmokeUniforms['u_gridSpacing'], GRID_SPACING);
      gl.uniform3fv(initializeSmokeUniforms['u_simulationSpace'], SIMULATION_SPACE.toArray());
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapSmokeFbObj();
    }

    const addBuoyancyForce = function(deltaTime) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, velocityFbObjW.framebuffer);
      gl.useProgram(addBuoyancyForceProgram);
      gl.uniform1i(addBuoyancyForceUniforms['u_cellNum'], CELL_NUM);
      gl.uniform1i(addBuoyancyForceUniforms['u_cellTextureSize'], cellTextureSize);
      setUniformTexture(gl, 0, velocityFbObjR.velocityTexture, addBuoyancyForceUniforms['u_velocityTexture']);
      setUniformTexture(gl, 1, smokeFbObjR.smokeTexture, addBuoyancyForceUniforms['u_smokeTexture']);
      gl.uniform1f(addBuoyancyForceUniforms['u_deltaTime'], deltaTime);
      gl.uniform1f(addBuoyancyForceUniforms['u_densityScale'], parameters['density force']);
      gl.uniform1f(addBuoyancyForceUniforms['u_temperatureScale'], parameters['temperature force']);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapVelocityFbObj();
    };

    const advectVelocity = function(deltaTime) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, velocityFbObjW.framebuffer);
      gl.useProgram(advectVelocityProgram);
      gl.uniform1i(advectVelocityUniforms['u_cellNum'], CELL_NUM);
      gl.uniform1i(advectVelocityUniforms['u_cellTextureSize'], cellTextureSize);
      gl.uniform3iv(advectVelocityUniforms['u_resolution'], SIMULATION_RESOLUTION.toArray());
      setUniformTexture(gl, 0, velocityFbObjR.velocityTexture, advectVelocityUniforms['u_velocityTexture']);
      gl.uniform1f(advectVelocityUniforms['u_deltaTime'], deltaTime);
      gl.uniform1f(advectVelocityUniforms['u_gridSpacing'], GRID_SPACING);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapVelocityFbObj();
    };

    const computePressure = function(deltaTime) {
      gl.useProgram(computePressureProgram);
      gl.uniform1i(computePressureUniforms['u_cellNum'], CELL_NUM);
      gl.uniform1i(computePressureUniforms['u_cellTextureSize'], cellTextureSize);
      gl.uniform3iv(computePressureUniforms['u_resolution'], SIMULATION_RESOLUTION.toArray());
      setUniformTexture(gl, 0, velocityFbObjR.velocityTexture, computePressureUniforms['u_velocityTexture']);
      gl.uniform1f(computePressureUniforms['u_deltaTime'], deltaTime);
      gl.uniform1f(computePressureUniforms['u_gridSpacing'], GRID_SPACING);
      gl.uniform1f(computePressureUniforms['u_density'], parameters['air density']);
      for (let i = 0; i < 10; i++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, pressureFbObjW.framebuffer);
        setUniformTexture(gl, 1, pressureFbObjR.pressureTexture, computePressureUniforms['u_pressureTexture'])
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        swapPressureFbObj();
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    };

    const addPressureForce = function(deltaTime) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, velocityFbObjW.framebuffer);
      gl.useProgram(addPressureForceProgram);
      gl.uniform1i(addPressureForceUniforms['u_cellNum'], CELL_NUM);
      gl.uniform1i(addPressureForceUniforms['u_cellTextureSize'], cellTextureSize);
      gl.uniform3iv(addPressureForceUniforms['u_resolution'], SIMULATION_RESOLUTION.toArray());
      setUniformTexture(gl, 0, velocityFbObjR.velocityTexture, addPressureForceUniforms['u_velocityTexture']);
      setUniformTexture(gl, 1, pressureFbObjR.pressureTexture, addPressureForceUniforms['u_pressureTexture']);
      gl.uniform1f(addPressureForceUniforms['u_deltaTime'], deltaTime);
      gl.uniform1f(addPressureForceUniforms['u_gridSpacing'], GRID_SPACING);
      gl.uniform1f(addPressureForceUniforms['u_density'], parameters['air density']);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapVelocityFbObj();
    };

    const decayVelocity = function(deltaTime) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, velocityFbObjW.framebuffer);
      gl.useProgram(decayVelocityProgram);
      setUniformTexture(gl, 0, velocityFbObjR.velocityTexture, decayVelocityUniforms['u_velocityTexture']);
      gl.uniform1f(decayVelocityUniforms['u_deltaTime'], deltaTime);
      gl.uniform1f(decayVelocityUniforms['u_velocityDecay'], parameters['velocity decay']);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapVelocityFbObj();
    }

    const updateVelocity = function(deltaTime) {
      addBuoyancyForce(deltaTime);
      advectVelocity(deltaTime);
      computePressure(deltaTime);
      addPressureForce(deltaTime);
      decayVelocity(deltaTime);
    };

    const advectSmoke = function(deltaTime) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, smokeFbObjW.framebuffer);
      gl.useProgram(advectSmokeProgram);
      gl.uniform1i(advectSmokeUniforms['u_cellNum'], CELL_NUM);
      gl.uniform1i(advectSmokeUniforms['u_cellTextureSize'], cellTextureSize);
      gl.uniform3iv(advectSmokeUniforms['u_resolution'], SIMULATION_RESOLUTION.toArray());
      setUniformTexture(gl, 0, velocityFbObjR.velocityTexture, advectSmokeUniforms['u_velocityTexture']);
      setUniformTexture(gl, 1, smokeFbObjR.smokeTexture, advectSmokeUniforms['u_smokeTexture']);
      gl.uniform1f(advectSmokeUniforms['u_deltaTime'], deltaTime);
      gl.uniform1f(advectSmokeUniforms['u_gridSpacing'], GRID_SPACING);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapSmokeFbObj();
    }

    const addSmoke = function(deltaTime) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, smokeFbObjW.framebuffer);
      gl.useProgram(addSmokeProgram);
      gl.uniform1i(addSmokeUniforms['u_cellNum'], CELL_NUM);
      gl.uniform1i(addSmokeUniforms['u_cellTextureSize'], cellTextureSize);
      gl.uniform3iv(addSmokeUniforms['u_resolution'], SIMULATION_RESOLUTION.toArray());
      gl.uniform3fv(addSmokeUniforms['u_simulationSpace'], SIMULATION_SPACE.toArray());
      setUniformTexture(gl, 0, smokeFbObjR.smokeTexture, addSmokeUniforms['u_smokeTexture']);
      gl.uniform1f(addSmokeUniforms['u_deltaTime'], deltaTime);
      gl.uniform1f(addSmokeUniforms['u_gridSpacing'], GRID_SPACING);
      gl.uniform1i(addSmokeUniforms['u_addHeat'], mousePressing);
      const heatSourceCenter = new Vector2(mousePosition.x / canvas.width, mousePosition.y / canvas.height);
      gl.uniform2fv(addSmokeUniforms['u_mousePosition'], heatSourceCenter.toArray());
      gl.uniform1f(addSmokeUniforms['u_heatSourceRadius'], parameters['heat radius']);
      gl.uniform1f(addSmokeUniforms['u_heatSourceIntensity'], parameters['heat intensity']);
      gl.uniform1f(addSmokeUniforms['u_densityDecay'], parameters['density decay']);
      gl.uniform1f(addSmokeUniforms['u_temperatureDecay'], parameters['temperature decay']);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapSmokeFbObj();
    }

    const updateSmoke = function(deltaTime) {
      advectSmoke(deltaTime);
      addSmoke(deltaTime);
    }

    const stepSimulation = function(deltaTime) {
      gl.viewport(0.0, 0.0, cellTextureSize, cellTextureSize);
      updateVelocity(deltaTime);
      updateSmoke(deltaTime);
    }


    const RENDER_SCALE = Vector3.mul(SIMULATION_SPACE, 75.0 / SIMULATION_SPACE.y);
    const CAMERA_POSITION = new Vector3(100.0, 100.0, 150.0);
    const MODEL_MATRIX = Matrix4.identity;
    const iNV_MODEL_MATRIX = Matrix4.identity; 
    const VIEW_MATRIX = Matrix4.inverse(Matrix4.lookAt(
      CAMERA_POSITION,
      Vector3.zero,
      new Vector3(0.0, 1.0, 0.0)
    ));
    let mvpMatrix;
    const setMvpMatrix = function() {
      const projectionMatrix = Matrix4.perspective(canvas.width / canvas.height, 60.0, 0.01, 1000.0);
      mvpMatrix = Matrix4.mul(VIEW_MATRIX, projectionMatrix);
    };
    setMvpMatrix();
    window.addEventListener('resize', _ => {
      setMvpMatrix();
    });

    const renderVelocity = function() {
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(renderVelocityProgram);
      gl.uniformMatrix4fv(renderVelocityUniforms['u_mvpMatrix'], false, mvpMatrix.elements);
      gl.uniformMatrix4fv(renderVelocityUniforms['u_modelMatrix'], false, MODEL_MATRIX.elements);
      gl.uniformMatrix4fv(renderVelocityUniforms['u_invModelMatrix'], false, iNV_MODEL_MATRIX.elements);
      gl.uniform3fv(renderVelocityUniforms['u_scale'], RENDER_SCALE.toArray());
      gl.uniform3fv(renderVelocityUniforms['u_cameraPosition'], CAMERA_POSITION.toArray());
      gl.uniform1i(renderVelocityUniforms['u_cellTextureSize'], cellTextureSize);
      gl.uniform3iv(renderVelocityUniforms['u_resolution'], SIMULATION_RESOLUTION.toArray());
      gl.uniform3fv(renderVelocityUniforms['u_simulationSpace'], SIMULATION_SPACE.toArray());
      setUniformTexture(gl, 0, velocityFbObjR.velocityTexture, renderVelocityUniforms['u_velocityTexture']);
      gl.uniform1f(renderVelocityUniforms['u_gridSpacing'], GRID_SPACING);
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.CULL_FACE);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.drawArrays(gl.TRIANGLES, 0, 36);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
      gl.disable(gl.BLEND);
    }

    const renderDensity = function() {
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(renderDensityProgram);
      gl.uniformMatrix4fv(renderDensityUniforms['u_mvpMatrix'], false, mvpMatrix.elements);
      gl.uniformMatrix4fv(renderDensityUniforms['u_modelMatrix'], false, MODEL_MATRIX.elements);
      gl.uniformMatrix4fv(renderDensityUniforms['u_invModelMatrix'], false, iNV_MODEL_MATRIX.elements);
      gl.uniform3fv(renderDensityUniforms['u_scale'], RENDER_SCALE.toArray());
      gl.uniform3fv(renderDensityUniforms['u_cameraPosition'], CAMERA_POSITION.toArray());
      gl.uniform1i(renderDensityUniforms['u_cellTextureSize'], cellTextureSize);
      gl.uniform3iv(renderDensityUniforms['u_resolution'], SIMULATION_RESOLUTION.toArray());
      gl.uniform3fv(renderDensityUniforms['u_simulationSpace'], SIMULATION_SPACE.toArray());
      setUniformTexture(gl, 0, smokeFbObjR.smokeTexture, renderDensityUniforms['u_smokeTexture']);
      gl.uniform1f(renderDensityUniforms['u_gridSpacing'], GRID_SPACING);
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.CULL_FACE);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.drawArrays(gl.TRIANGLES, 0, 36);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
      gl.disable(gl.BLEND);
    }

    const renderTemperature = function() {
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(renderTemperatureProgram);
      gl.uniformMatrix4fv(renderTemperatureUniforms['u_mvpMatrix'], false, mvpMatrix.elements);
      gl.uniformMatrix4fv(renderTemperatureUniforms['u_modelMatrix'], false, MODEL_MATRIX.elements);
      gl.uniformMatrix4fv(renderTemperatureUniforms['u_invModelMatrix'], false, iNV_MODEL_MATRIX.elements);
      gl.uniform3fv(renderTemperatureUniforms['u_scale'], RENDER_SCALE.toArray());
      gl.uniform3fv(renderTemperatureUniforms['u_cameraPosition'], CAMERA_POSITION.toArray());
      gl.uniform1i(renderTemperatureUniforms['u_cellTextureSize'], cellTextureSize);
      gl.uniform3iv(renderTemperatureUniforms['u_resolution'], SIMULATION_RESOLUTION.toArray());
      gl.uniform3fv(renderTemperatureUniforms['u_simulationSpace'], SIMULATION_SPACE.toArray());
      setUniformTexture(gl, 0, smokeFbObjR.smokeTexture, renderTemperatureUniforms['u_smokeTexture']);
      gl.uniform1f(renderTemperatureUniforms['u_gridSpacing'], GRID_SPACING);
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.CULL_FACE);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.drawArrays(gl.TRIANGLES, 0, 36);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
      gl.disable(gl.BLEND);
    }

    const render = function() {
      gl.viewport(0.0, 0.0, canvas.width, canvas.height);
      if (parameters['render'] === 'velocity') {
        renderVelocity();
      } else if (parameters['render'] === 'temperature') {
        renderTemperature();
      } else {
        renderDensity();
      }
    };

    initializeVelocity();
    initializeSmoke();
    let simulationSeconds = 0.0;
    let remaindedSimulationSeconds = 0.0;
    let previousRealSeconds = performance.now() * 0.001;
    const loop = function() {
      stats.update();

      const currentRealSeconds = performance.now() * 0.001;
      const nextSimulationSeconds = simulationSeconds + remaindedSimulationSeconds + parameters['time scale'] * Math.min(0.02, currentRealSeconds - previousRealSeconds);
      previousRealSeconds = currentRealSeconds;
      const timeStep = parameters['time step'];
      while(nextSimulationSeconds - simulationSeconds > timeStep) {
        stepSimulation(timeStep);
        simulationSeconds += timeStep;
      }
      remaindedSimulationSeconds = nextSimulationSeconds - simulationSeconds;

      render();
      requestId = requestAnimationFrame(loop);
    };
    loop();
  };
  reset();

}());