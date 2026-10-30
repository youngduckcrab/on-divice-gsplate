#version 450
// Gaussian falloff: the quad's local coordinates are scaled so that
// exp(-|local|^2) equals exp(-0.5 * Mahalanobis distance^2).
layout(location = 0) in vec2 v_local;
layout(location = 1) in vec4 v_color;

layout(location = 0) out vec4 out_color;

void main() {
  float alpha = min(v_color.a * exp(-dot(v_local, v_local)), 0.99);
  if (alpha < 1.0 / 255.0) discard;
  out_color = vec4(v_color.rgb * alpha, alpha);
}
