class Vector2 {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  static get zero() {
    return new Vector2(0.0, 0.0);
  }

  static add(v1, v2) {
    return new Vector2(v1.x + v2.x, v1.y + v2.y);
  }

  static sub(v1, v2) {
    return new Vector2(v1.x - v2.x, v1.y - v2.y);
  }

  static mul(v, s) {
    return new Vector2(v.x * s, v.y * s);
  }

  static div(v, s) {
    return new Vector2(v.x / s, v.y / s);
  }

  static norm(v) {
    const m = v.mag();
    return new Vector2(v.x / m, v.y / m);
  }

  static dot(v1, v2) {
    return v1.x * v2.x + v1.y * v2.y;
  }

  static dist(v1, v2) {
    const v = Vector2.sub(v1, v2);
    return v.mag();
  }

  static equals(v1, v2) {
    return v1.x === v2.x && v1.y === v2.y;
  }

  add(v) {
    this.x += v.x;
    this.y += v.y;
    return this;
  }

  sub(v) {
    this.x -= v.x;
    this.y -= v.y;
    return this;
  }

  mul(s) {
    this.x *= s;
    this.y *= s;
    return this;
  }

  div(s) {
    this.x /= s;
    this.y /= s;
    return this;
  }

  ceil() {
    this.x = Math.ceil(this.x);
    this.y = Math.ceil(this.y);
    return this;
  }

  mag() {
    return Math.sqrt(this.sqMag());
  }

  sqMag() {
    return this.x * this.x + this.y * this.y;
  }

  norm() {
    return Vector2.norm(this);
  }

  dot(v) {
    return Vector3.dot(this, v);
  }

  toArray() {
    return [this.x, this.y];
  }
}

class Vector3 {
  constructor(x, y, z) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  static get zero() {
    return new Vector3(0.0, 0.0, 0.0);
  }

  static add(v1, v2) {
    return new Vector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
  }

  static sub(v1, v2) {
    return new Vector3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
  }

  static mul(v, s) {
    return new Vector3(v.x * s, v.y * s, v.z * s);
  }

  static div(v, s) {
    return new Vector3(v.x / s, v.y / s, v.z / s);
  }

  static norm(v) {
    const m = v.mag();
    return new Vector3(v.x / m, v.y / m, v.z / m);
  }

  static dot(v1, v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  }

  static cross(v1, v2) {
    return new Vector3(
      v1.y * v2.z - v1.z * v2.y,
      v1.z * v2.x - v1.x * v2.z,
      v1.x * v2.y - v1.y * v2.x 
    );
  }

  static dist(v1, v2) {
    const v = Vector3.sub(v1, v2);
    return v.mag();
  }

  add(v) {
    this.x += v.x;
    this.y += v.y;
    this.z += v.z;
    return this;
  }

  sub(v) {
    this.x -= v.x;
    this.y -= v.y;
    this.z -= v.z;
    return this;
  }

  mul(s) {
    this.x *= s;
    this.y *= s;
    this.z *= s;
    return this;
  }

  div(s) {
    this.x /= s;
    this.y /= s;
    this.z /= s;
    return this;
  }

  ceil() {
    this.x = Math.ceil(this.x);
    this.y = Math.ceil(this.y);
    this.z = Math.ceil(this.z);
    return this;
  }

  mag() {
    return Math.sqrt(this.sqMag());
  }

  sqMag() {
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }

  norm() {
    const m = this.mag();
    this.div(m);
    return this;
  }

  dot(v) {
    return Vector3.dot(this, v);
  }

  cross(v) {
    return Vector3.cross(this, v);
  }

  toArray() {
    return [this.x, this.y, this.z];
  }
}

class Matrix4 {
  constructor(elements) {
    this.elements = new Float32Array(elements);
  }

  get determinant() {
    const e = this.elements;

    return e[0] * (e[5] * e[10] * e[15] + e[9] * e[14] * e[7] + e[13] * e[6] * e[11] - e[13] * e[10] * e[7] - e[9] * e[6] * e[15] - e[5] * e[14] * e[11])
         - e[4] * (e[1] * e[10] * e[15] + e[9] * e[14] * e[3] + e[13] * e[2] * e[11] - e[13] * e[10] * e[3] - e[9] * e[2] * e[15] - e[1] * e[14] * e[11])
         + e[8] * (e[1] * e[6] * e[15] + e[5] * e[14] * e[3] + e[13] * e[2] * e[7] - e[13] * e[6] * e[3] - e[5] * e[2] * e[15] - e[1] * e[14] * e[7])
         - e[8] * (e[1] * e[6] * e[11] + e[5] * e[10] * e[3] + e[9] * e[2] * e[7] - e[9] * e[6] * e[3] - e[5] * e[2] * e[11] - e[1] * e[10] * e[7])
  }

  static get identity() {
    return new Matrix4([
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0
    ]);
  }

  static translate(x, y, z) {
    return new Matrix4([
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      x, y, z, 1.0
    ]);
  }

  static scale(x, y, z) {
    return new Matrix4([
      x, 0.0, 0.0, 0.0,
      0.0, y, 0.0, 0.0,
      0.0, 0.0, z, 0.0,
      0.0, 0.0, 0.0, 1.0
    ]);
  }

  static rotateX(r) {
    const c = Math.cos(r);
    const s = Math.sin(r);
    return new Matrix4([
      1.0, 0.0, 0.0, 0.0,
      0.0, c, s, 0.0,
      0.0, -s, c, 0.0,
      0.0, 0.0, 0.0, 1.0
    ]);
  }

  static rotateY(r) {
    const c = Math.cos(r);
    const s = Math.sin(r);
    return new Matrix4([
      c, 0.0, -s, 0.0,
      0.0, 1.0, 0.0, 0.0,
      s, 0.0, c, 0.0,
      0.0, 0.0, 0.0, 1.0
    ]);
  }

  static rotateZ(r) {
    const c = Math.cos(r);
    const s = Math.sin(r);
    return new Matrix4([
      c, s, 0.0, 0.0,
      -s, c, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0
    ]);
  }

  static rotateXYZ(x, y, z) {
    const cx = Math.cos(x);
    const sx = Math.sin(x);
    const cy = Math.cos(y);
    const sy = Math.sin(y);
    const cz = Math.cos(z);
    const sz = Math.sin(z);
    return new Matrix4([
      cy * cz, cy * sz, -sy, 0.0,
      sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy, 0.0,
      cx * sy * cz + sx * sz, cx * sy * sz - sx * cz, cx * cy, 0.0,
      0.0, 0.0, 0.0, 1.0
    ]);
  }

  static transpose(m) {
    const e = m.elements;
    return new Matrix4([
      e[0], e[4], e[8], e[12],
      e[1], e[5], e[9], e[13],
      e[2], e[6], e[10], e[14],
      e[3], e[7], e[11], e[15]
    ]);
  }

  static inverse(m) {
    const d = m.determinant;
    if (Math.abs(d) <= 0.0) {
      throw new Error('not invertiable');
    }
    const invD = 1.0 / d;
    const e = m.elements;

    return new Matrix4([
      (e[5] * e[10] * e[15] + e[9] * e[14] * e[7] + e[13] * e[6] * e[11]
        - e[13] * e[10] * e[7] - e[9] * e[6] * e[15] - e[5] * e[14] * e[11]) * invD,
      -(e[1] * e[10] * e[15] + e[9] * e[14] * e[3] + e[13] * e[2] * e[11]
        - e[13] * e[10] * e[3] - e[9] * e[2] * e[15] - e[1] * e[14] * e[11]) * invD,
      (e[1] * e[6] * e[15] + e[5] * e[14] * e[3] + e[13] * e[2] * e[7]
        - e[13] * e[6] * e[3] - e[5] * e[2] * e[15] - e[1] * e[14] * e[7]) * invD,
      -(e[1] * e[6] * e[11] + e[5] * e[10] * e[3] + e[9] * e[2] * e[7]
        - e[9] * e[6] * e[3] - e[5] * e[2] * e[11] - e[1] * e[10] * e[7]) * invD,
      -(e[4] * e[10] * e[15] + e[8] * e[14] * e[7] + e[12] * e[6] * e[11]
        - e[12] * e[10] * e[7] - e[8] * e[6] * e[15] - e[4] * e[14] * e[11]) * invD,
      (e[0] * e[10] * e[15] + e[8] * e[14] * e[3] + e[12] * e[2] * e[11]
        - e[12] * e[10] * e[3] - e[8] * e[2] * e[15] - e[0] * e[14] * e[11]) * invD,
      -(e[0] * e[6] * e[15] + e[4] * e[14] * e[3] + e[12] * e[2] * e[7]
        - e[12] * e[6] * e[3] - e[4] * e[2] * e[15] - e[0] * e[14] * e[7]) * invD,
      (e[0] * e[6] * e[11] + e[4] * e[10] * e[3] + e[8] * e[2] * e[7]
        - e[8] * e[6] * e[3] - e[4] * e[2] * e[11] - e[0] * e[10] * e[7]) * invD,
      (e[4] * e[9] * e[15] + e[8] * e[13] * e[7] + e[12] * e[5] * e[11]
        - e[12] * e[9] * e[7] - e[8] * e[5] * e[15] - e[4] * e[13] * e[11]) * invD,
      -(e[0] * e[9] * e[15] + e[8] * e[13] * e[3] + e[12] * e[1] * e[11]
        - e[12] * e[9] * e[3] - e[8] * e[1] * e[15] - e[0] * e[13] * e[11]) * invD,
      (e[0] * e[5] * e[15] + e[4] * e[13] * e[3] + e[12] * e[1] * e[7]
        - e[12] * e[5] * e[3] - e[4] * e[1] * e[15] - e[0] * e[13] * e[7]) * invD,
      -(e[0] * e[5] * e[11] + e[4] * e[9] * e[3] + e[8] * e[1] * e[7]
        - e[8] * e[5] * e[3] - e[4] * e[1] * e[11] - e[0] * e[9] * e[7]) * invD,
      -(e[4] * e[9] * e[14] + e[8] * e[13] * e[6] + e[12] * e[5] * e[10]
        - e[12] * e[9] * e[6] - e[8] * e[5] * e[14] - e[4] * e[13] * e[10]) * invD,
      (e[0] * e[9] * e[14] + e[8] * e[13] * e[2] + e[12] * e[1] * e[10]
        - e[12] * e[9] * e[2] - e[8] * e[1] * e[14] - e[0] * e[13] * e[10]) * invD,
      -(e[0] * e[5] * e[14] + e[4] * e[13] * e[2] + e[12] * e[1] * e[6]
        - e[12] * e[5] * e[2] - e[4] * e[1] * e[14] - e[0] * e[13] * e[6]) * invD,
      (e[0] * e[5] * e[10] + e[4] * e[9] * e[2] + e[8] * e[1] * e[6]
        - e[8] * e[5] * e[2] - e[4] * e[1] * e[10] - e[0] * e[9] * e[6]) * invD
    ]);
  }

  static mul(m1, m2) {
    const e1 = m1.elements;
    const e2 = m2.elements;
    return new Matrix4([
      e1[0] * e2[0] + e1[1] * e2[4] + e1[2] * e2[8] + e1[3] * e2[12],
      e1[0] * e2[1] + e1[1] * e2[5] + e1[2] * e2[9] + e1[3] * e2[13],
      e1[0] * e2[2] + e1[1] * e2[6] + e1[2] * e2[10] + e1[3] * e2[14],
      e1[0] * e2[3] + e1[1] * e2[7] + e1[2] * e2[11] + e1[3] * e2[15],

      e1[4] * e2[0] + e1[5] * e2[4] + e1[6] * e2[8] + e1[7] * e2[12],
      e1[4] * e2[1] + e1[5] * e2[5] + e1[6] * e2[9] + e1[7] * e2[13],
      e1[4] * e2[2] + e1[5] * e2[6] + e1[6] * e2[10] + e1[7] * e2[14],
      e1[4] * e2[3] + e1[5] * e2[7] + e1[6] * e2[11] + e1[7] * e2[15],

      e1[8] * e2[0] + e1[9] * e2[4] + e1[10] * e2[8] + e1[11] * e2[12],
      e1[8] * e2[1] + e1[9] * e2[5] + e1[10] * e2[9] + e1[11] * e2[13],
      e1[8] * e2[2] + e1[9] * e2[6] + e1[10] * e2[10] + e1[11] * e2[14],
      e1[8] * e2[3] + e1[9] * e2[7] + e1[10] * e2[11] + e1[11] * e2[15],

      e1[12] * e2[0] + e1[13] * e2[4] + e1[14] * e2[8] + e1[15] * e2[12],
      e1[12] * e2[1] + e1[13] * e2[5] + e1[14] * e2[9] + e1[15] * e2[13],
      e1[12] * e2[2] + e1[13] * e2[6] + e1[14] * e2[10] + e1[15] * e2[14],
      e1[12] * e2[3] + e1[13] * e2[7] + e1[14] * e2[11] + e1[15] * e2[15],
    ]);
  }

  static perspective(aspect, vfov, near, far) {
    const theta = vfov * Math.PI / 180.0;
    const t = near * Math.tan(theta * 0.5);
    const r = aspect * t;
    const fpn = far + near;
    const fmn = far - near;

    return new Matrix4([
      near / r, 0.0, 0.0, 0.0,
      0.0, near / t, 0.0, 0.0,
      0.0, 0.0, -fpn / fmn, -1.0,
      0.0, 0.0, -2.0 * far * near / fmn, 0.0
    ]);
  }

  static orthographic(width, height, near, far) {
    const invFmn = 1.0 / (far - near);
    return new Matrix4([
      2.0 / width, 0.0, 0.0, 0.0,
      0.0, 2.0 / height, 0.0, 0.0,
      0.0, 0.0, -2.0 * invFmn, 0.0,
      0.0, 0.0, -(far + near) * invFmn, 1.0 
    ]);
  }

  static lookAt(origin , target, up) {
    const front = Vector3.sub(target, origin).norm();
    const z = Vector3.mul(front, -1);
    const x = Vector3.cross(up, z);
    const y = Vector3.cross(z, x);

    return new Matrix4([
      x.x, x.y, x.z, 0.0,
      y.x, y.y, y.z, 0.0,
      z.x, z.y, z.z, 0.0,
      origin.x, origin.y, origin.z, 1.0
    ]);
  }
}