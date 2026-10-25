#version 330 core
//#include common

// Full-screen analytic ray tracing of suns, planets, moons and rings.
// Output is premultiplied colour + coverage; depth is written so stars are occluded.

#define MAXB 16
in vec2 v_uv;
out vec4 f_color;

uniform vec2 u_tan;
uniform vec2 u_res;
uniform float u_time;
uniform float u_logfar;
uniform int u_count;
uniform vec4 u_pos[MAXB];    // centre (camera space), radius
uniform mat3 u_rot[MAXB];    // body-local -> camera rotation
uniform vec4 u_type[MAXB];   // type, seed, ring inner, ring outer (in radii)
uniform vec4 u_c0[MAXB];
uniform vec4 u_c1[MAXB];
uniform vec4 u_c2[MAXB];
uniform vec4 u_atm[MAXB];    // atmosphere rgb, thickness (fraction of radius)
uniform vec4 u_misc[MAXB];   // fade, p1, p2, p3 (type specific)
uniform vec4 u_ring[MAXB];   // ring rgb, opacity
uniform int u_nsun;
uniform vec4 u_sunPos[2];    // camera-space centre, radius
uniform vec4 u_sunCol[2];    // irradiance colour

const float AMBIENT = 0.004;

// --------------------------------------------------------------- helpers
vec3 rotY(vec3 p, float a) {
    float c = cos(a), s = sin(a);
    return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

vec3 rotateAxis(vec3 p, vec3 a, float ang) {
    float c = cos(ang), s = sin(ang);
    return p * c + cross(a, p) * s + a * dot(a, p) * (1.0 - c);
}

// fbm with level-of-detail: octaves smaller than ~a pixel (fp = pixel size in p units) fade out
float fbmL(vec3 p, int oct, float fp) {
    float sum = 0.0, amp = 0.5, freq = 1.0;
    for (int k = 0; k < 9; k++) {
        if (k >= oct) break;
        float w = clamp(1.6 - 2.0 * fp * freq, 0.0, 1.0);
        if (w <= 0.0) break;
        sum += amp * w * snoise(p * freq + float(k) * vec3(13.1, 7.7, 3.3));
        freq *= 2.03;
        amp *= 0.5;
    }
    return sum;
}

float craterField(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    float h = 0.0;
    vec3 sg = step(0.5, f) * 2.0 - 1.0;   // only the 2x2x2 neighbourhood on f's side
    for (int z = 0; z <= 1; z++)
    for (int y = 0; y <= 1; y++)
    for (int x = 0; x <= 1; x++) {
        vec3 o = vec3(float(x), float(y), float(z)) * sg;
        vec3 rnd = hash33(i + o);
        if (rnd.z < 0.35) continue;
        vec3 cpos = o + 0.2 + 0.6 * hash33(i + o + 5.31);
        float rad = 0.14 + 0.26 * rnd.x * rnd.x;
        float d = length(f - cpos) / rad;
        if (d < 1.45) {
            float bowl = d < 1.0 ? (d * d - 1.0) : 0.0;
            float rim = 0.28 * exp(-pow((d - 1.0) / 0.2, 2.0));
            h += (bowl + rim) * rad;
        }
    }
    return h;
}

void tangents(vec3 n, out vec3 t1, out vec3 t2) {
    vec3 up = abs(n.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    t1 = normalize(cross(up, n));
    t2 = cross(n, t1);
}

vec3 bumpNormal(vec3 lp, float h0, float h1, float h2, vec3 t1, vec3 t2, float e, float k) {
    vec3 g = ((h1 - h0) * t1 + (h2 - h0) * t2) / e;
    return normalize(lp - k * g);
}

// --------------------------------------------------------------- shadows
float occlusion(vec3 P, vec3 L, int self, float sunAng) {
    float sh = 1.0;
    for (int i = 0; i < MAXB; i++) {
        if (i >= u_count) break;
        if (i == self || u_type[i].x < 0.5) continue;
        vec3 c = u_pos[i].xyz - P;
        float r = u_pos[i].w;
        float tc = dot(c, L);
        if (tc <= 0.0) continue;
        float d = length(c - L * tc);
        float pen = max(tc * sunAng, 1e-3 * r);
        sh *= smoothstep(-pen, pen, d - r);
    }
    return sh;
}

float ringAlpha(int i, float rr, float fp) {
    float ri = u_type[i].z, ro = u_type[i].w;
    if (rr < ri || rr > ro) return 0.0;
    float x = (rr - ri) / (ro - ri);
    float seed = u_type[i].y;
    float w1 = clamp(1.5 - fp * 35.0 * 2.0, 0.0, 1.0);
    float w2 = clamp(1.5 - fp * 140.0 * 2.0, 0.0, 1.0);
    float d = 0.55 + 0.35 * snoise(vec3(x * 7.0, seed, 0.0))
            + w1 * 0.25 * snoise(vec3(x * 35.0, seed, 3.0))
            + w2 * 0.18 * snoise(vec3(x * 140.0, seed, 7.0));
    d *= smoothstep(0.0, 0.05, x) * smoothstep(1.0, 0.9, x);
    float gap = fract(seed * 7.13) * 0.4 + 0.35;
    d *= 1.0 - 0.92 * smoothstep(0.03, 0.012, abs(x - gap));
    return clamp(d, 0.0, 1.0) * u_ring[i].w;
}

float ringShadow(int i, vec3 P, vec3 L) {
    if (u_type[i].z <= 0.0) return 1.0;
    vec3 nrm = u_rot[i][1];
    float dn = dot(L, nrm);
    if (abs(dn) < 1e-4) return 1.0;
    float t = dot(u_pos[i].xyz - P, nrm) / dn;
    if (t <= 0.0) return 1.0;
    float rr = length(P + L * t - u_pos[i].xyz) / u_pos[i].w;
    return 1.0 - 0.85 * ringAlpha(i, rr, 0.01);
}

// --------------------------------------------------------------- surfaces
struct Surf {
    vec3 albedo;
    vec3 emis;     // always visible (lava)
    vec3 night;    // only on night side (city lights)
    float spec;
    vec3 nrm;      // local-space perturbed normal
    float cloud;
};

float terrainH(vec3 lp, float seed, float fp) { return fbmL(lp * 1.7 + seed, 8, fp * 1.7); }

float rockyH(vec3 lp, float seed, float fp) {
    float h = 0.55 * fbmL(lp * 2.2 + seed, 6, fp * 2.2);
    float w1 = clamp(1.5 - fp * 5.0 * 2.0, 0.0, 1.0);
    if (w1 > 0.0) h += 0.22 * w1 * craterField(lp * 5.0 + seed);
    float w2 = clamp(1.5 - fp * 14.0 * 2.0, 0.0, 1.0);
    if (w2 > 0.0) h += 0.08 * w2 * craterField(lp * 14.0 + seed * 1.31);
    return h;
}

Surf surface(int i, int type, vec3 lp, float fp) {
    Surf s;
    s.albedo = vec3(0.5); s.emis = vec3(0.0); s.night = vec3(0.0);
    s.spec = 0.0; s.nrm = lp; s.cloud = 0.0;
    float seed = u_type[i].y;
    vec3 c0 = u_c0[i].rgb, c1 = u_c1[i].rgb, c2 = u_c2[i].rgb;
    vec4 m = u_misc[i];
    vec3 t1, t2;
    tangents(lp, t1, t2);
    float e = max(fp, 0.0015);

    if (type == 1) {                                   // terran
        float h = terrainH(lp, seed, fp);
        float sea = m.y;
        float land = smoothstep(sea - 0.004, sea + 0.004, h);
        float depth = clamp((sea - h) / 0.35, 0.0, 1.0);
        vec3 ocean = mix(c0 * 2.2 + vec3(0.0, 0.035, 0.04), c0, sqrt(depth));
        float hl = clamp((h - sea) / 0.45, 0.0, 1.0);
        float moist = fbmL(lp * 3.1 + seed + 4.0, 4, fp * 3.1) * 0.5 + 0.5;
        vec3 ground = mix(c1, c1 * vec3(1.6, 1.25, 0.8) + vec3(0.07, 0.05, 0.0), smoothstep(0.5, 0.68, moist));
        ground = mix(ground, c2, smoothstep(0.25, 0.8, hl));
        ground = mix(ground, vec3(0.8, 0.82, 0.86), smoothstep(0.8, 0.95, hl));
        float lat = abs(lp.y) + 0.08 * fbmL(lp * 5.0 + seed, 3, fp * 5.0);
        float ice = smoothstep(0.80, 0.84, lat + 0.15 * hl);
        s.albedo = mix(mix(ocean, ground, land), vec3(0.85, 0.9, 0.95), ice);
        s.spec = (1.0 - land) * (1.0 - ice);
        if (land > 0.0) {
            float h1 = terrainH(normalize(lp + e * t1), seed, fp);
            float h2 = terrainH(normalize(lp + e * t2), seed, fp);
            s.nrm = normalize(mix(lp, bumpNormal(lp, h, h1, h2, t1, t2, e, 0.06), land));
        }
        vec3 cp = rotY(lp, u_time * 0.004);
        vec3 wq = vec3(fbmL(cp * 2.0 + seed + 20.0, 3, fp * 2.0), fbmL(cp * 2.0 + seed + 30.0, 3, fp * 2.0), 0.0);
        float cn = fbmL(cp * 2.6 + seed + 40.0 + 0.7 * wq, 6, fp * 2.6);
        s.cloud = smoothstep(0.0, 0.32, cn + m.z - 0.5) * 0.95;
        float city = land * (1.0 - ice) * m.w;
        if (city > 0.0) {
            float cw = clamp(1.5 - fp * 40.0 * 2.0, 0.0, 1.0);
            float fine = mix(0.5, snoise(lp * 40.0 + seed) * 0.5 + 0.5, cw);
            float region = fbmL(lp * 6.0 + seed + 50.0, 3, fp * 6.0) * 0.5 + 0.5;
            city *= smoothstep(0.62, 0.9, fine * 0.55 + region * 0.65) * smoothstep(0.02, 0.1, h - sea);
            s.night = vec3(1.0, 0.68, 0.32) * city * 2.0;
        }
    } else if (type == 2) {                            // gas giant
        float bandF = m.y;
        float ang = u_time * 0.006 * (1.0 + 0.6 * sin(lp.y * 9.0 + seed));
        vec3 p = rotY(lp, ang);
        float sl = m.z;
        float so = seed * 2.7;
        vec3 sdir = vec3(cos(so) * sqrt(1.0 - sl * sl), sl, sin(so) * sqrt(1.0 - sl * sl));
        float sd = length(p - sdir);
        float srad = 0.08 + 0.06 * fract(seed * 3.3);
        float swirl = exp(-sd * sd / (srad * srad));
        p = rotateAxis(p, sdir, swirl * 3.5);
        float tur = fbmL(p * vec3(3.0, 9.0, 3.0) + seed, 6, fp * 9.0);
        float y = p.y + tur * 0.045 * m.w;
        float b1 = sin(y * bandF + seed) * 0.5 + 0.5;
        float b2 = sin(y * bandF * 2.37 + seed * 1.9) * 0.5 + 0.5;
        float b3 = fbmL(vec3(y * 18.0, seed, 0.5), 3, fp * 18.0) * 0.5 + 0.5;
        vec3 col = mix(c0, c1, smoothstep(0.25, 0.75, b1));
        col = mix(col, c2, smoothstep(0.5, 0.95, b2) * 0.7);
        col *= 0.75 + 0.5 * b3;
        float fine = fbmL(vec3(p.x * 6.0, y * 70.0, p.z * 6.0) + seed, 4, fp * 70.0);
        col *= 0.92 + 0.18 * fine;
        col = mix(col, c2 * vec3(1.15, 0.72, 0.55), smoothstep(0.25, 0.9, swirl) * 0.75 * step(0.35, fract(seed * 7.7)));
        col *= mix(1.0, 0.72, smoothstep(0.7, 0.97, abs(lp.y)));
        s.albedo = col;
    } else if (type == 3) {                            // ice
        float n = fbmL(lp * 2.5 + seed, 6, fp * 2.5);
        float w1 = clamp(1.5 - fp * 6.0 * 2.0, 0.0, 1.0);
        float w2 = clamp(1.5 - fp * 18.0 * 2.0, 0.0, 1.0);
        float cr = w1 * pow(1.0 - abs(snoise(lp * 6.0 + seed + 0.4 * n)), 14.0)
                 + 0.6 * w2 * pow(1.0 - abs(snoise(lp * 18.0 + seed * 1.7)), 18.0);
        s.albedo = mix(c2, c0, smoothstep(-0.5, 0.6, n));
        s.albedo = mix(s.albedo, c1, clamp(cr, 0.0, 1.0) * 0.8);
        s.spec = 0.25;
        float n1 = fbmL(normalize(lp + e * t1) * 2.5 + seed, 6, fp * 2.5);
        float n2 = fbmL(normalize(lp + e * t2) * 2.5 + seed, 6, fp * 2.5);
        s.nrm = bumpNormal(lp, n, n1, n2, t1, t2, e, 0.015);
    } else if (type == 4) {                            // lava
        float n = fbmL(lp * 2.0 + seed, 6, fp * 2.0);
        float r1 = 1.0 - abs(snoise(lp * 4.0 + seed + 0.5 * vec3(n)));
        float w2 = clamp(1.5 - fp * 11.0 * 2.0, 0.0, 1.0);
        float r2 = 1.0 - abs(snoise(lp * 11.0 + seed * 1.3));
        float cracks = pow(r1, 22.0) + w2 * 0.5 * pow(r2, 26.0);
        float lakes = smoothstep(-0.33, -0.5, n);
        float lava = clamp(cracks + lakes, 0.0, 1.0);
        s.albedo = mix(c0, c1, smoothstep(-0.3, 0.5, n)) * (1.0 - lava);
        float pulse = 0.8 + 0.2 * sin(u_time * 0.7 + n * 12.0);
        s.emis = c2 * lava * pulse * mix(0.8, 3.5, lava);
        float n1 = fbmL(normalize(lp + e * t1) * 2.0 + seed, 6, fp * 2.0);
        float n2 = fbmL(normalize(lp + e * t2) * 2.0 + seed, 6, fp * 2.0);
        s.nrm = bumpNormal(lp, n, n1, n2, t1, t2, e, 0.04);
    } else {                                           // rocky (5) / barren moon (6)
        float h0 = rockyH(lp, seed, fp);
        float h1 = rockyH(normalize(lp + e * t1), seed, fp);
        float h2 = rockyH(normalize(lp + e * t2), seed, fp);
        s.nrm = bumpNormal(lp, h0, h1, h2, t1, t2, e, 0.12);
        float n = fbmL(lp * 3.0 + seed + 7.0, 5, fp * 3.0);
        vec3 col = mix(c0, c1, smoothstep(-0.4, 0.4, n));
        col = mix(col, c2, smoothstep(-0.05, -0.3, h0) * 0.7);
        col *= 0.8 + 0.4 * clamp(h0 * 2.0 + 0.5, 0.0, 1.0);
        if (type == 5 && m.y > 0.5)
            col = mix(col, vec3(0.85, 0.85, 0.88), smoothstep(0.86, 0.9, abs(lp.y) + 0.06 * n));
        s.albedo = col;
    }
    return s;
}

// --------------------------------------------------------------- shading
vec4 shadeSun(int i, vec3 rd, float t, float pa) {
    vec3 c = u_pos[i].xyz;
    float r = u_pos[i].w;
    vec3 P = rd * t;
    vec3 N = normalize(P - c);
    float tc = dot(c, rd);
    float dd = length(c - rd * tc);
    float cov = clamp((r - dd) / (tc * pa) + 0.5, 0.0, 1.0);
    vec3 lp = transpose(u_rot[i]) * N;
    float fp = t * pa / r;
    float seed = u_type[i].y;
    float mu = max(dot(N, -rd), 0.0);
    float limb = 0.3 + 0.7 * pow(mu, 0.55);
    float g = fbmL(lp * 22.0 + seed + vec3(0.0, u_time * 0.015, 0.0), 5, fp * 22.0);
    float cell = 1.0 - abs(fbmL(lp * 60.0 + seed + vec3(u_time * 0.03), 3, fp * 60.0));
    float spots = smoothstep(0.62, 0.72, snoise(lp * 3.5 + seed) * 0.5 + 0.5) * smoothstep(0.55, 0.1, abs(lp.y));
    vec3 col = u_c0[i].rgb * u_c0[i].w * limb * (0.82 + 0.25 * g + 0.12 * cell) * (1.0 - 0.75 * spots);
    col = mix(col, col * vec3(1.0, 0.55, 0.3), (1.0 - mu) * 0.55);
    return vec4(col * cov, cov);
}

vec4 shadeSphere(int i, vec3 rd, float t, float pa) {
    int type = int(u_type[i].x + 0.5);
    if (type == 0) return shadeSun(i, rd, t, pa);

    vec3 c = u_pos[i].xyz;
    float r = u_pos[i].w;
    vec3 P = rd * t;
    vec3 N = normalize(P - c);
    vec3 V = -rd;
    float tc = dot(c, rd);
    float dd = length(c - rd * tc);
    float cov = clamp((r - dd) / (tc * pa) + 0.5, 0.0, 1.0);
    mat3 R = u_rot[i];
    vec3 lp = transpose(R) * N;
    float fp = t * pa / r / max(dot(N, V), 0.15);

    Surf s = surface(i, type, lp, fp);
    vec3 Nb = normalize(R * s.nrm);

    vec3 diff = vec3(0.0), spec = vec3(0.0), cloudL = vec3(0.0), atmL = vec3(0.0), sunset = vec3(0.0);
    float day = 0.0;
    for (int j = 0; j < 2; j++) {
        if (j >= u_nsun) break;
        vec3 Ls = u_sunPos[j].xyz - P;
        float Ld = length(Ls);
        vec3 L = Ls / Ld;
        float sunAng = u_sunPos[j].w / Ld;
        float sh = occlusion(P, L, i, sunAng) * ringShadow(i, P, L);
        vec3 sc = u_sunCol[j].rgb * sh;
        float ng = dot(N, L);
        float term = smoothstep(-0.03, 0.12, ng);
        float ndl = type == 2 ? max((dot(Nb, L) + 0.05) / 1.05, 0.0) : max(dot(Nb, L), 0.0);
        diff += sc * ndl * term;
        vec3 H = normalize(L + V);
        float nh = max(dot(N, H), 0.0);
        spec += sc * (pow(nh, 70.0) * 0.4 + pow(nh, 900.0) * 3.0) * term;
        cloudL += sc * clamp(ng * 1.1 + 0.05, 0.0, 1.0);
        atmL += sc * smoothstep(-0.25, 0.35, ng);
        sunset += sc * exp(-ng * ng * 25.0) * step(-0.25, ng);
        day = max(day, smoothstep(-0.12, 0.12, ng) * sh);
    }

    vec3 col = s.albedo * (diff + AMBIENT)
             + spec * s.spec * (1.0 - s.cloud)
             + s.emis * (1.0 - 0.7 * s.cloud)
             + s.night * (1.0 - day) * (1.0 - 0.85 * s.cloud);
    col = mix(col, vec3(0.92) * cloudL + AMBIENT, s.cloud);

    float th = u_atm[i].w;
    if (th > 0.0) {
        float den = clamp(th / 0.03, 0.0, 1.6);
        float mu = max(dot(N, V), 0.0);
        float rim = pow(1.0 - mu, 3.0);
        vec3 ac = u_atm[i].rgb;
        col = col * (1.0 - 0.45 * rim * den) + ac * atmL * (0.05 + 1.25 * rim) * den;
        col += vec3(1.0, 0.42, 0.12) * sunset * rim * 0.5 * den;
    }
    return vec4(col * cov, cov);
}

vec4 shadeRing(int i, vec3 rd, float t, float pa) {
    vec3 c = u_pos[i].xyz;
    float r = u_pos[i].w;
    vec3 P = rd * t;
    vec3 nrm = u_rot[i][1];
    float rr = length(P - c) / r;
    float dn = abs(dot(rd, nrm));
    float fp = t * pa / max(dn, 0.02) / r / (u_type[i].w - u_type[i].z);
    float a = ringAlpha(i, rr, fp);
    a = 1.0 - pow(1.0 - min(a, 0.995), 1.0 / max(dn, 0.08));
    float x = (rr - u_type[i].z) / (u_type[i].w - u_type[i].z);
    vec3 col = u_ring[i].rgb * (0.75 + 0.45 * snoise(vec3(x * 11.0, u_type[i].y, 11.0)));
    col = mix(col, u_c2[i].rgb, 0.25 * smoothstep(0.3, 0.9, snoise(vec3(x * 4.0, u_type[i].y, 21.0))));
    vec3 light = vec3(0.0);
    float vs = dot(-rd, nrm);
    for (int j = 0; j < 2; j++) {
        if (j >= u_nsun) break;
        vec3 Ls = u_sunPos[j].xyz - P;
        float Ld = length(Ls);
        vec3 L = Ls / Ld;
        float sh = occlusion(P, L, -1, u_sunPos[j].w / Ld);
        float ls = dot(L, nrm);
        float lit = (ls * vs > 0.0) ? 0.25 + 0.75 * abs(ls) : 0.45 * (1.0 - a) + 0.05;
        float fwd = pow(max(dot(rd, L), 0.0), 8.0) * 1.5;
        light += u_sunCol[j].rgb * sh * lit * (1.0 + fwd);
    }
    return vec4(col * (light + AMBIENT) * a, a);
}

// --------------------------------------------------------------- glows
vec3 bodyGlow(int i, vec3 rd, out float tcOut) {
    vec3 c = u_pos[i].xyz;
    float r = u_pos[i].w;
    float tc = dot(c, rd);
    tcOut = tc;
    if (tc <= 0.0) return vec3(0.0);
    vec3 v = rd * tc - c;
    float dd = length(v);
    int type = int(u_type[i].x + 0.5);
    if (type == 0) {
        float b = max(dd / r, 1.0);
        float k = b - 1.0;
        vec3 dirL = normalize(transpose(u_rot[i]) * v + 1e-6);
        float seed = u_type[i].y;
        float s1 = snoise(dirL * 2.5 + vec3(seed, u_time * 0.02, 0.0)) * 0.5 + 0.5;
        float s2 = snoise(dirL * 7.0 + vec3(seed * 1.3, 0.0, u_time * 0.03)) * 0.5 + 0.5;
        float streamers = 0.4 + 1.0 * s1 * s2;
        vec3 col = u_c0[i].rgb * u_c0[i].w;
        float g = 0.30 * exp(-k * 10.0) * (0.6 + 0.8 * s2)
                + 0.05 * exp(-k * 2.0) * streamers
                + 0.010 * exp(-k * 0.45)
                + 0.0012 / (b * b);
        return col * g;
    }
    float th = u_atm[i].w;
    if (th <= 0.0 || dd <= r || dd >= r * (1.0 + th)) return vec3(0.0);
    float x = (dd - r) / (r * th);
    float dens = exp(-x * 4.0) * (1.0 - x);
    vec3 n = v / dd;
    vec3 lit = vec3(0.0);
    for (int j = 0; j < 2; j++) {
        if (j >= u_nsun) break;
        vec3 L = normalize(u_sunPos[j].xyz - c);
        lit += u_sunCol[j].rgb * (smoothstep(-0.3, 0.4, dot(n, L)) + 2.5 * pow(max(dot(rd, L), 0.0), 12.0));
    }
    return u_atm[i].rgb * dens * lit * 1.3 * clamp(th / 0.03, 0.0, 1.6);
}

// --------------------------------------------------------------- main
void insertHit(inout float ht[4], inout int hid[4], inout int hk[4], inout int n, float t, int id, int kind) {
    if (n == 4 && t >= ht[3]) return;
    int p = min(n, 3);
    while (p > 0 && ht[p - 1] > t) {
        ht[p] = ht[p - 1]; hid[p] = hid[p - 1]; hk[p] = hk[p - 1];
        p--;
    }
    ht[p] = t; hid[p] = id; hk[p] = kind;
    n = min(n + 1, 4);
}

void main() {
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec3 rd = normalize(vec3(ndc * u_tan, 1.0));
    float pa = 2.0 * u_tan.y / u_res.y;

    float ht[4] = float[4](1e30, 1e30, 1e30, 1e30);
    int hid[4] = int[4](0, 0, 0, 0);
    int hk[4] = int[4](0, 0, 0, 0);
    int n = 0;

    for (int i = 0; i < MAXB; i++) {
        if (i >= u_count) break;
        vec3 c = u_pos[i].xyz;
        float r = u_pos[i].w;
        float tc = dot(c, rd);
        if (tc > 0.0) {
            float dd = length(c - rd * tc);
            if ((r - dd) / (tc * pa) + 0.5 > 0.0) {
                float h = sqrt(max(r * r - dd * dd, 0.0));
                float t = tc - h;
                if (t > 0.0) insertHit(ht, hid, hk, n, t, i, 0);
            }
        }
        if (u_type[i].z > 0.0) {
            vec3 nrm = u_rot[i][1];
            float dn = dot(rd, nrm);
            if (abs(dn) > 1e-6) {
                float t = dot(c, nrm) / dn;
                if (t > 0.0) {
                    float rr = length(rd * t - c) / r;
                    if (rr > u_type[i].z && rr < u_type[i].w) {
                        // the planet itself hides the far side of the ring
                        float dd = length(c - rd * tc);
                        float h = sqrt(max(r * r - dd * dd, 0.0));
                        if (!(dd < r && t > tc - h)) insertHit(ht, hid, hk, n, t, i, 1);
                    }
                }
            }
        }
    }

    vec3 acc = vec3(0.0);
    float trans = 1.0;
    float depthT = -1.0;
    float la[4] = float[4](0.0, 0.0, 0.0, 0.0);
    for (int k = 0; k < 4; k++) {
        if (k >= n || trans < 0.002) break;
        int i = hid[k];
        vec4 s = hk[k] == 0 ? shadeSphere(i, rd, ht[k], pa) : shadeRing(i, rd, ht[k], pa);
        s *= u_misc[i].x;
        acc += trans * s.rgb;
        la[k] = s.a;
        trans *= 1.0 - s.a;
        if (depthT < 0.0 && trans < 0.5) depthT = ht[k];
    }

    for (int i = 0; i < MAXB; i++) {
        if (i >= u_count) break;
        float tc;
        vec3 g = bodyGlow(i, rd, tc);
        if (dot(g, g) <= 0.0) continue;
        float tr = 1.0;
        for (int k = 0; k < 4; k++) {
            if (k >= n) break;
            if (ht[k] < tc) tr *= 1.0 - la[k];
        }
        acc += tr * g * u_misc[i].x;
    }

    f_color = vec4(acc, 1.0 - trans);
    // stars are infinitely far away: anything solid hides them
    gl_FragDepth = depthT < 0.0 ? 1.0 : 0.0;
}
