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
    const velocityTexture = createTexture(gl, width, height, gl.RG32F, gl.RG, gl.FLOAT);
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

out vec2 o_velocity;

void main(void) {
  o_velocity = vec2(0.0);
}
`;

  const INITIALIZE_SMOKE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec2 o_smoke; // x: density, y: temperature

#define AMBIENT_TEMPERATURE 273.0

void main(void) {
  float density = 0.0;
  float temperature = AMBIENT_TEMPERATURE;
  o_smoke = vec2(density, temperature);
}
`;

  const ADD_BUOYANCY_FORCE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec2 o_velocity;

uniform sampler2D u_velocityTexture;
uniform sampler2D u_smokeTexture;
uniform float u_deltaTime;
uniform vec2 u_gravity;
uniform float u_densityScale;
uniform float u_temperatureScale;

#define GRAVITY vec2(0.0, -9.8)
#define AMBIENT_TEMPERATURE 273.0

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec2 velocity = texelFetch(u_velocityTexture, coord, 0).xy;
  vec2 smoke = texelFetch(u_smokeTexture, coord, 0).xy;
  vec2 buoyancy = (u_densityScale * smoke.x
    - u_temperatureScale * (smoke.y - AMBIENT_TEMPERATURE)) * GRAVITY;
  o_velocity = velocity + u_deltaTime * buoyancy;
}
`;

  const ADVECT_VELOCITY_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec2 o_velocity;

uniform sampler2D u_velocityTexture;
uniform float u_deltaTime;
uniform float u_gridSpacing;

vec2 sampleVelocity(ivec2 coord, ivec2 textureSize) {
  vec2 velocity = texelFetch(u_velocityTexture, coord, 0).xy;
  if (coord.x < 0 || coord.x >= textureSize.x) {
    velocity.x = 0.0;
  }
  if (coord.y < 0 || coord.y >= textureSize.y) {
    velocity.y = 0.0;
  }
  return velocity;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec2 position = (vec2(coord) + 0.5) * u_gridSpacing;
  vec2 velocity = texelFetch(u_velocityTexture, coord, 0).xy;

  vec2 prevPos = position - u_deltaTime * velocity;
  vec2 prevCoord = prevPos / u_gridSpacing - 0.5;

  ivec2 i = ivec2(prevCoord);
  vec2 f = fract(prevCoord);

  ivec2 textureSize = textureSize(u_velocityTexture, 0);
  vec2 vel00 = sampleVelocity(i, textureSize);
  vec2 vel10 = sampleVelocity(i + ivec2(1, 0), textureSize);
  vec2 vel01 = sampleVelocity(i + ivec2(0, 1), textureSize);
  vec2 vel11 = sampleVelocity(i + ivec2(1, 1), textureSize);

  o_velocity = mix(mix(vel00, vel10, f.x), mix(vel01, vel11, f.x), f.y);
}
`;

  const COMPUTE_PRESSURE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out float o_pressure;

uniform sampler2D u_velocityTexture;
uniform sampler2D u_pressureTexture;
uniform float u_deltaTime;
uniform float u_gridSpacing;
uniform float u_density;

vec2 sampleVelocity(ivec2 coord, ivec2 textureSize) {
  vec2 velocity = texelFetch(u_velocityTexture, coord, 0).xy;
  if (coord.x < 0 || coord.x >= textureSize.x) {
    velocity.x = 0.0;
  }
  if (coord.y < 0 || coord.y >= textureSize.y) {
    velocity.y = 0.0;
  }
  return velocity;
}

float samplePressure(ivec2 coord, ivec2 textureSize) {
  if (coord.x < 0) {
    coord.x = 0;
  }
  if (coord.x >= textureSize.x) {
    coord.x = textureSize.x - 1;
  }
  if (coord.y < 0) {
    coord.y = 0;
  }
  if (coord.y >= textureSize.y) {
    coord.y = textureSize.y - 1;
  }
  return texelFetch(u_pressureTexture, coord, 0).x;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);

  ivec2 pressTexSize = textureSize(u_pressureTexture, 0);
  float pl = samplePressure(coord + ivec2(-1, 0), pressTexSize);
  float pr = samplePressure(coord + ivec2(1, 0), pressTexSize);
  float pd = samplePressure(coord + ivec2(0, -1), pressTexSize);
  float pu = samplePressure(coord + ivec2(0, 1), pressTexSize);

  ivec2 velTexSize = textureSize(u_velocityTexture, 0);
  vec2 vl = sampleVelocity(coord + ivec2(-1, 0), velTexSize);
  vec2 vr = sampleVelocity(coord + ivec2(1, 0), velTexSize);
  vec2 vd = sampleVelocity(coord + ivec2(0, -1), velTexSize);
  vec2 vu = sampleVelocity(coord + ivec2(0, 1), velTexSize);

  o_pressure = 0.25 * (pl + pr + pd + pu
    - 0.5 * (vr.x - vl.x + vu.y - vd.y) * u_gridSpacing * u_density / u_deltaTime);
}
`;

  const ADD_PRESSURE_FORCE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec2 o_velocity;

uniform sampler2D u_velocityTexture;
uniform sampler2D u_pressureTexture;
uniform float u_deltaTime;
uniform float u_gridSpacing;
uniform float u_density;

float samplePressure(ivec2 coord, ivec2 textureSize) {
  if (coord.x < 0) {
    coord.x = 0;
  }
  if (coord.x >= textureSize.x) {
    coord.x = textureSize.x - 1;
  }
  if (coord.y < 0) {
    coord.y = 0;
  }
  if (coord.y >= textureSize.y) {
    coord.y = textureSize.y - 1;
  }
  return texelFetch(u_pressureTexture, coord, 0).x;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec2 velocity = texelFetch(u_velocityTexture, coord, 0).xy;

  ivec2 pressTexSize = textureSize(u_pressureTexture, 0);
  float pl = samplePressure(coord + ivec2(-1, 0), pressTexSize);
  float pr = samplePressure(coord + ivec2(1, 0), pressTexSize);
  float pd = samplePressure(coord + ivec2(0, -1), pressTexSize);
  float pu = samplePressure(coord + ivec2(0, 1), pressTexSize);

  o_velocity = velocity - 0.5 * u_deltaTime * vec2(pr - pl, pu - pd) / (u_gridSpacing * u_density);
}
`;

  const DECAY_VELOCITY_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec2 o_velocity;

uniform sampler2D u_velocityTexture;
uniform float u_deltaTime;
uniform float u_velocityDecay;

void main(void) {
  vec2 velocity = texelFetch(u_velocityTexture, ivec2(gl_FragCoord.xy), 0).xy;
  velocity *= exp(-u_velocityDecay * u_deltaTime);
  o_velocity = velocity;
}
`;

  const ADVECT_SMOKE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec2 o_smoke;

uniform sampler2D u_velocityTexture;
uniform sampler2D u_smokeTexture;
uniform float u_deltaTime;
uniform float u_gridSpacing;

#define AMBIENT_TEMPERATURE 273.0

vec2 sampleSmoke(ivec2 coord, ivec2 textureSize) {
  vec2 smoke = texelFetch(u_smokeTexture, coord, 0).xy;
  if (coord.x < 0 || coord.x >= textureSize.x || coord.y < 0 || coord.y >= textureSize.y) {
    smoke = vec2(0.0, AMBIENT_TEMPERATURE);
  }
  return smoke;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec2 position = (vec2(coord) + 0.5) * u_gridSpacing;
  vec2 velocity = texelFetch(u_velocityTexture, coord, 0).xy;

  vec2 prevPos = position - u_deltaTime * velocity;
  vec2 prevCoord = prevPos / u_gridSpacing - 0.5;

  ivec2 i = ivec2(prevCoord);
  vec2 f = fract(prevCoord);

  ivec2 textureSize = textureSize(u_smokeTexture, 0);
  vec2 smoke00 = sampleSmoke(i, textureSize);
  vec2 smoke10 = sampleSmoke(i + ivec2(1, 0), textureSize);
  vec2 smoke01 = sampleSmoke(i + ivec2(0, 1), textureSize);
  vec2 smoke11 = sampleSmoke(i + ivec2(1, 1), textureSize);

  o_smoke = mix(mix(smoke00, smoke10, f.x), mix(smoke01, smoke11, f.x), f.y);
}
`;

  const ADD_SMOKE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec2 o_smoke;

uniform sampler2D u_smokeTexture;
uniform float u_deltaTime;
uniform float u_gridSpacing;
uniform bool u_addHeat;
uniform vec2 u_heatSourceCenter;
uniform float u_heatSourceRadius;
uniform float u_heatSourceIntensity;
uniform float u_densityDecay;
uniform float u_temperatureDecay;

#define AMBIENT_TEMPERATURE 273.0

void main(void) {
  vec2 smoke = texelFetch(u_smokeTexture, ivec2(gl_FragCoord), 0).xy;
  float density = smoke.x;
  float temperature = smoke.y;

  float nextTemperature = temperature;
  vec2 position = (floor(gl_FragCoord.xy) + 0.5) * u_gridSpacing;
  if (u_addHeat) {
    nextTemperature += smoothstep(u_heatSourceRadius, 0.0, length(position - u_heatSourceCenter))
      * u_deltaTime * u_heatSourceIntensity;
  }
  nextTemperature += (1.0 - exp(-u_temperatureDecay * u_deltaTime)) * (AMBIENT_TEMPERATURE - nextTemperature);

  float nextDensity = density + u_deltaTime * max(0.0, (nextTemperature - (AMBIENT_TEMPERATURE + 100.0))) * 0.01;
  nextDensity *= exp(-u_densityDecay * u_deltaTime);

  o_smoke = vec2(nextDensity, nextTemperature);
}

`;

  const RENDER_VELOCITY_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

#define PI 3.14159265359

uniform sampler2D u_velocityTexture;

out vec4 o_color;

vec3 hsv2rgb(float h, float s, float v) {
  h = mod(h, 360.0);
  if (s == 0.0) {
    return vec3(0.0, 0.0, 0.0);
  }
  float c = v * s;
  float i = h / 60.0;
  float x = c * (1.0 - abs(mod(i, 2.0) - 1.0)); 
  return vec3(v - c) + (i < 1.0 ? vec3(c, x, 0.0) : 
    i < 2.0 ? vec3(x, c, 0.0) :
    i < 3.0 ? vec3(0.0, c, x) :
    i < 4.0 ? vec3(0.0, x, c) :
    i < 5.0 ? vec3(x, 0.0, c) :
    vec3(c, 0.0, x));
}

void main(void) {
  vec2 velocity = texelFetch(u_velocityTexture, ivec2(gl_FragCoord.xy), 0).xy;

  vec2 normVel = normalize(velocity);
  float radian = atan(velocity.y, velocity.x) + PI;

  float hue = 360.0 * radian / (2.0 * PI);
  float brightness = min(1.0, length(velocity) / 0.5);
  vec3 color = hsv2rgb(hue, 1.0, brightness);

  o_color = vec4(color, 1.0);
}
`;

  const RENDER_DENSITY_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec4 o_color;

uniform sampler2D u_smokeTexture;

void main(void) {
  float density = texelFetch(u_smokeTexture, ivec2(gl_FragCoord.xy), 0).x;
  o_color = vec4(vec3(density), 1.0);
}
`;

const RENDER_TEMPERATURE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec4 o_color;

uniform sampler2D u_smokeTexture;

#define AMBIENT_TEMPERATURE 273.0

const vec4[6] TEMPERATURE_COLOR = vec4[](
  vec4(0.0, 0.0, 0.0, 0.0),
  vec4(0.0, 0.0, 0.0, AMBIENT_TEMPERATURE),
  vec4(1.0, 0.0, 0.0, AMBIENT_TEMPERATURE + 100.0),
  vec4(1.0, 0.5, 0.0, AMBIENT_TEMPERATURE + 200.0),
  vec4(1.0, 1.0, 1.0, AMBIENT_TEMPERATURE + 300.0),
  vec4(0.5, 0.5, 1.0, AMBIENT_TEMPERATURE + 400.0)
);

void main(void) {
  float temperature = texelFetch(u_smokeTexture, ivec2(gl_FragCoord.xy), 0).y;

  vec3 color = TEMPERATURE_COLOR[5].xyz;
  for (int i = 0; i < 5; i++) {
    if (temperature < TEMPERATURE_COLOR[i + 1].w) {
      color = mix(TEMPERATURE_COLOR[i].xyz, TEMPERATURE_COLOR[i + 1].xyz,
        1.0 - (TEMPERATURE_COLOR[i + 1].w - temperature) / (TEMPERATURE_COLOR[i + 1]. w - TEMPERATURE_COLOR[i].w));
      break;
    }
  }
  o_color = vec4(color, 1.0);
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
    'grid spacing': 0.001,
    'air density': 2.354, 
    'density force': 0.01,
    'temperature force': 0.0001,
    'heat radius': 0.1,
    'heat intensity': 1000.0,
    'velocity decay': 0.1,
    'density decay': 0.5,
    'temperature decay': 1.0,
    'time step': 0.005,
    'time scale': 1.0,
    'render': 'density',
    'reset': _ => reset()
  };

  const gui = new dat.GUI();
  gui.add(parameters, 'density force', 0.0, 0.1).step(0.0001);
  gui.add(parameters, 'temperature force', 0.0, 0.0003).step(0.00001);
  gui.add(parameters, 'heat radius', 0.0, 0.3).step(0.001);
  gui.add(parameters, 'heat intensity', 0.0, 2000.0).step(1.0);
  gui.add(parameters, 'velocity decay', 0.0, 5.0).step(0.1);
  gui.add(parameters, 'density decay', 0.0, 5.0).step(0.1);
  gui.add(parameters, 'temperature decay', 0.0, 5.0).step(0.1);
  gui.add(parameters, 'time step', 0.0001, 0.01).step(0.0001);
  gui.add(parameters, 'time scale', 0.5, 2.0).step(0.001);
  gui.add(parameters, 'render', ['density', 'temperature', 'velocity']);
  gui.add(parameters, 'reset');

  const canvas = document.getElementById('canvas');
  const gl = canvas.getContext('webgl2');
  gl.getExtension('EXT_color_buffer_float');

  const initializeVelocityProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, INITIALIZE_VELOCITY_FRAGMENT_SHADER_SOURCE);
  const initializeSmokeProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, INITIALIZE_SMOKE_FRAGMENT_SHADER_SOURCE);
  const addBuoyancyForceProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, ADD_BUOYANCY_FORCE_FRAGMENT_SHADER_SOURCE);
  const advectVelocityProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, ADVECT_VELOCITY_FRAGMENT_SHADER_SOURCE);
  const computePressureProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, COMPUTE_PRESSURE_FRAGMENT_SHADER_SOURCE);
  const addPressureForceProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, ADD_PRESSURE_FORCE_FRAGMENT_SHADER_SOURCE);
  const decayVelocityProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, DECAY_VELOCITY_FRAGMENT_SHADER_SOURCE);
  const advectSmokeProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, ADVECT_SMOKE_FRAGMENT_SHADER_SOURCE);
  const addSmokeProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, ADD_SMOKE_FRAGMENT_SHADER_SOURCE);
  const renderVelocityProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, RENDER_VELOCITY_FRAGMENT_SHADER_SOURCE);
  const renderDensityProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, RENDER_DENSITY_FRAGMENT_SHADER_SOURCE);
  const renderTemperatureProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, RENDER_TEMPERATURE_FRAGMENT_SHADER_SOURCE);

  const addBuoyancyForceUniforms = getUniformLocations(gl, addBuoyancyForceProgram, ['u_velocityTexture', 'u_smokeTexture', 'u_deltaTime', 'u_densityScale', 'u_temperatureScale']);
  const advectVelocityUniforms = getUniformLocations(gl, advectVelocityProgram, ['u_velocityTexture', 'u_deltaTime', 'u_gridSpacing']);
  const computePressureUniforms = getUniformLocations(gl, computePressureProgram, ['u_velocityTexture', 'u_pressureTexture', 'u_deltaTime', 'u_gridSpacing', 'u_density']);
  const addPressureForceUniforms = getUniformLocations(gl, addPressureForceProgram, ['u_velocityTexture', 'u_pressureTexture', 'u_deltaTime', 'u_gridSpacing', 'u_density']);
  const decayVelocityUniforms = getUniformLocations(gl, decayVelocityProgram, ['u_velocityTexture', 'u_deltaTime', 'u_velocityDecay']);
  const advectSmokeUniforms = getUniformLocations(gl, advectSmokeProgram, ['u_velocityTexture', 'u_smokeTexture', 'u_deltaTime', 'u_gridSpacing']);
  const addSmokeUniforms = getUniformLocations(gl, addSmokeProgram,
    ['u_smokeTexture', 'u_deltaTime', 'u_gridSpacing', 'u_addHeat', 'u_heatSourceCenter', 'u_heatSourceRadius', 'u_heatSourceIntensity', 'u_densityDecay', 'u_temperatureDecay']);
  const renderVelocityUniforms = getUniformLocations(gl, renderVelocityProgram, ['u_velocityTexture']);
  const renderDensityUniforms = getUniformLocations(gl, renderDensityProgram, ['u_smokeTexture']);
  const renderTemperatureUniforms = getUniformLocations(gl, renderTemperatureProgram, ['u_smokeTexture']);

  let requestId = null;
  const reset = function() {
    if (requestId !== null) {
      cancelAnimationFrame(requestId);
      requestId = null;
    }

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    gl.viewport(0.0, 0.0, canvas.width, canvas.height);

    let velocityFbObjR = createVelocityFramebuffer(gl, canvas.width, canvas.height);
    let velocityFbObjW = createVelocityFramebuffer(gl, canvas.width, canvas.height);
    const swapVelocityFbObj = function() {
      const tmp = velocityFbObjR;
      velocityFbObjR = velocityFbObjW;
      velocityFbObjW = tmp;
    };

    let pressureFbObjR = createPressureFramebuffer(gl, canvas.width, canvas.height);
    let pressureFbObjW = createPressureFramebuffer(gl, canvas.width, canvas.height);
    const swapPressureFbObj = function() {
      const tmp = pressureFbObjR;
      pressureFbObjR = pressureFbObjW;
      pressureFbObjW = tmp;
    };


    let smokeFbObjR = createSmokeFramebuffer(gl, canvas.width, canvas.height);
    let smokeFbObjW = createSmokeFramebuffer(gl, canvas.width, canvas.height);
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
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapSmokeFbObj();
    }

    const addBuoyancyForce = function(deltaTime) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, velocityFbObjW.framebuffer);
      gl.useProgram(addBuoyancyForceProgram);
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
      setUniformTexture(gl, 0, velocityFbObjR.velocityTexture, advectVelocityUniforms['u_velocityTexture']);
      gl.uniform1f(advectVelocityUniforms['u_deltaTime'], deltaTime);
      gl.uniform1f(advectVelocityUniforms['u_gridSpacing'], parameters['grid spacing']);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapVelocityFbObj();
    };

    const computePressure = function(deltaTime) {
      gl.useProgram(computePressureProgram);
      setUniformTexture(gl, 0, velocityFbObjR.velocityTexture, computePressureUniforms['u_velocityTexture']);
      gl.uniform1f(computePressureUniforms['u_deltaTime'], deltaTime);
      gl.uniform1f(computePressureUniforms['u_gridSpacing'], parameters['grid spacing']);
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
      setUniformTexture(gl, 0, velocityFbObjR.velocityTexture, addPressureForceUniforms['u_velocityTexture']);
      setUniformTexture(gl, 1, pressureFbObjR.pressureTexture, addPressureForceUniforms['u_pressureTexture']);
      gl.uniform1f(addPressureForceUniforms['u_deltaTime'], deltaTime);
      gl.uniform1f(addPressureForceUniforms['u_gridSpacing'], parameters['grid spacing']);
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
      setUniformTexture(gl, 0, velocityFbObjR.velocityTexture, advectSmokeUniforms['u_velocityTexture']);
      setUniformTexture(gl, 1, smokeFbObjR.smokeTexture, advectSmokeUniforms['u_smokeTexture']);
      gl.uniform1f(advectSmokeUniforms['u_deltaTime'], deltaTime);
      gl.uniform1f(advectSmokeUniforms['u_gridSpacing'], parameters['grid spacing']);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapSmokeFbObj();
    }

    const addSmoke = function(deltaTime) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, smokeFbObjW.framebuffer);
      gl.useProgram(addSmokeProgram);
      setUniformTexture(gl, 0, smokeFbObjR.smokeTexture, addSmokeUniforms['u_smokeTexture']);
      gl.uniform1f(addSmokeUniforms['u_deltaTime'], deltaTime);
      gl.uniform1f(addSmokeUniforms['u_gridSpacing'], parameters['grid spacing']);
      gl.uniform1i(addSmokeUniforms['u_addHeat'], mousePressing);
      const heatSourceCenter = Vector2.mul(mousePosition, parameters['grid spacing']);
      gl.uniform2fv(addSmokeUniforms['u_heatSourceCenter'], heatSourceCenter.toArray());
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
      updateVelocity(deltaTime);
      updateSmoke(deltaTime);
    }

    const renderVelocity = function() {
      gl.useProgram(renderVelocityProgram);
      setUniformTexture(gl, 0, velocityFbObjR.velocityTexture, renderVelocityUniforms['u_velocityTexture'])
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    const renderDensity = function() {
      gl.useProgram(renderDensityProgram);
      setUniformTexture(gl, 0, smokeFbObjR.smokeTexture, renderDensityUniforms['u_smokeTexture']);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    const renderTemperature = function() {
      gl.useProgram(renderTemperatureProgram);
      setUniformTexture(gl, 0, smokeFbObjR.smokeTexture, renderTemperatureUniforms['u_smokeTexture']);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    const render = function() {
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

  window.addEventListener('resize', _ => {
    reset();
  });

}());