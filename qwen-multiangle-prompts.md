# Prompt Format

`<sks> [azimuth] [elevation] [distance]`

# 96 Camera Positions

4 Elevations × 8 Azimuths × 3 Distances = 96 Poses

## Azimuths (Horizontal Rotation)
```
                         0° 
                    (front view)
                         │
         315°            │            45°
    (front-left)         │       (front-right)
              ╲          │          ╱
               ╲         │         ╱
                ╲        │        ╱
   270° ─────────────── ● ─────────────── 90°
   (left side)        OBJECT         (right side)
                ╱        │        ╲
               ╱         │         ╲
              ╱          │          ╲
         225°            │            135°
     (back-left)         │       (back-right)
                         │
                        180°
                    (back view)
```

### AngleDescriptors

0°: `front view`

45°: `front-right quarter view`

90°: `right side view`

135°: `back-right quarter view`

180°: `back view`

225°: `back-left quarter view`

270°: `left side view`

315°: `front-left quarter view`.

## Elevations (Vertical Angle)

AngleDescriptorDescription-30°low-angle shotCamera below, looking up0°eye-level shotCamera at object level30°elevated shotCamera slightly above60°high-angle shotCamera high, looking down

## Distances

FactorDescriptorUsage×0.6close-upDetails, textures×1.0medium shotBalanced, standard×1.8wide shotContext, environment

# All 96 Prompts Reference

## CLOSE-UP (32 prompts)

### Low-angle (-30°)

`<sks> front view low-angle shot close-up`

`<sks> front-right quarter view low-angle shot close-up`

`<sks> right side view low-angle shot close-up`

`<sks> back-right quarter view low-angle shot close-up`

`<sks> back view low-angle shot close-up`

`<sks> back-left quarter view low-angle shot close-up`

`<sks> left side view low-angle shot close-up`

`<sks> front-left quarter view low-angle shot close-up`

### Eye-level (0°)

`<sks> front view eye-level shot close-up`

`<sks> front-right quarter view eye-level shot close-up`

`<sks> right side view eye-level shot close-up`

`<sks> back-right quarter view eye-level shot close-up`

`<sks> back view eye-level shot close-up`

`<sks> back-left quarter view eye-level shot close-up`

`<sks> left side view eye-level shot close-up`

`<sks> front-left quarter view eye-level shot close-up`

### Elevated (30°)

`<sks> front view elevated shot close-up`

`<sks> front-right quarter view elevated shot close-up`

`<sks> right side view elevated shot close-up`

`<sks> back-right quarter view elevated shot close-up`

`<sks> back view elevated shot close-up`

`<sks> back-left quarter view elevated shot close-up`

`<sks> left side view elevated shot close-up`

`<sks> front-left quarter view elevated shot close-up`

### High-angle (60°)

`<sks> front view high-angle shot close-up`

`<sks> front-right quarter view high-angle shot close-up`

`<sks> right side view high-angle shot close-up`

`<sks> back-right quarter view high-angle shot close-up`

`<sks> back view high-angle shot close-up`

`<sks> back-left quarter view high-angle shot close-up`

`<sks> left side view high-angle shot close-up`

`<sks> front-left quarter view high-angle shot close-up`

## MEDIUM SHOT (32 prompts)

### Low-angle (-30°)

`<sks> front view low-angle shot medium shot`

`<sks> front-right quarter view low-angle shot medium shot`

`<sks> right side view low-angle shot medium shot`

`<sks> back-right quarter view low-angle shot medium shot`

`<sks> back view low-angle shot medium shot`

`<sks> back-left quarter view low-angle shot medium shot`

`<sks> left side view low-angle shot medium shot`

`<sks> front-left quarter view low-angle shot medium shot`

### Eye-level (0°) — Reference pose: front view eye-level shot medium shot

`<sks> front view eye-level shot medium shot`

`<sks> front-right quarter view eye-level shot medium shot`

`<sks> right side view eye-level shot medium shot`

`<sks> back-right quarter view eye-level shot medium shot`

`<sks> back view eye-level shot medium shot`

`<sks> back-left quarter view eye-level shot medium shot`

`<sks> left side view eye-level shot medium shot`

`<sks> front-left quarter view eye-level shot medium shot`

### Elevated (30°)

`<sks> front view elevated shot medium shot`

`<sks> front-right quarter view elevated shot medium shot`

`<sks> right side view elevated shot medium shot`

`<sks> back-right quarter view elevated shot medium shot`

`<sks> back view elevated shot medium shot`

`<sks> back-left quarter view elevated shot medium shot`

`<sks> left side view elevated shot medium shot`

`<sks> front-left quarter view elevated shot medium shot`

### High-angle (60°)

`<sks> front view high-angle shot medium shot`

`<sks> front-right quarter view high-angle shot medium shot`

`<sks> right side view high-angle shot medium shot`

`<sks> back-right quarter view high-angle shot medium shot`

`<sks> back view high-angle shot medium shot`

`<sks> back-left quarter view high-angle shot medium shot`

`<sks> left side view high-angle shot medium shot`

`<sks> front-left quarter view high-angle shot medium shot`

## WIDE SHOT (32 prompts)

### Low-angle (-30°)

`<sks> front view low-angle shot wide shot`

`<sks> front-right quarter view low-angle shot wide shot`

`<sks> right side view low-angle shot wide shot`

`<sks> back-right quarter view low-angle shot wide shot`

`<sks> back view low-angle shot wide shot`

`<sks> back-left quarter view low-angle shot wide shot`

`<sks> left side view low-angle shot wide shot`

`<sks> front-left quarter view low-angle shot wide shot`

### Eye-level (0°)

`<sks> front view eye-level shot wide shot`

`<sks> front-right quarter view eye-level shot wide shot`

`<sks> right side view eye-level shot wide shot`

`<sks> back-right quarter view eye-level shot wide shot`

`<sks> back view eye-level shot wide shot`

`<sks> back-left quarter view eye-level shot wide shot`

`<sks> left side view eye-level shot wide shot`

`<sks> front-left quarter view eye-level shot wide shot`

### Elevated (30°)

`<sks> front view elevated shot wide shot`

`<sks> front-right quarter view elevated shot wide shot`

`<sks> right side view elevated shot wide shot`

`<sks> back-right quarter view elevated shot wide shot`

`<sks> back view elevated shot wide shot`

`<sks> back-left quarter view elevated shot wide shot`

`<sks> left side view elevated shot wide shot`

`<sks> front-left quarter view elevated shot wide shot`

### High-angle (60°)


`<sks> front view high-angle shot wide shot`

`<sks> front-right quarter view high-angle shot wide shot`

`<sks> right side view high-angle shot wide shot`

`<sks> back-right quarter view high-angle shot wide shot`

`<sks> back view high-angle shot wide shot`

`<sks> back-left quarter view high-angle shot wide shot`

`<sks> left side view high-angle shot wide shot`

`<sks> front-left quarter view high-angle shot wide shot`