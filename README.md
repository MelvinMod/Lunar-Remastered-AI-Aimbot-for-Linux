# Lunar-Remastered-AI-Aimbot-for-Linux
I'm MelvinSGjr, the first(?) creator of [ai aimbot for linux](https://github.com/MelvinSGjr/Lunar-AI-Aimbot-for-Linux/tree/main), I have lost access to aupload anything on my previous account, so I made a new one

## Key Differences Summary:

### **1. Architecture & Modularity**
- **New `aimbot.py`**: Modular OOP design with separate classes:
  - `MousePathRecorder` - Records and replays mouse paths
  - `OptimizedAimingController` - Handles aiming logic
  - `ProAimEnhancer` - Adds professional aiming techniques
  - `Aimbot` - Main orchestrator
- **Old `aimbot (copy).py`**: Single monolithic class with mixed responsibilities

### **2. Aiming Philosophy**
- **New**: "Your style + Pro enhancement" - Keeps your natural aiming style while adding professional techniques
- **Old**: Complex learning system with pattern matching and adaptive parameters

### **3. Movement System**
- **New**: Records and replays YOUR actual mouse movements for natural-looking aim
- **Old**: Calculates movements based on mathematical formulas and learned patterns

### **4. Speed & Performance**
- **New**: Optimized for 240 FPS with minimal processing overhead
- **Old**: Variable FPS with complex calculations

### **5. Key Features Missing in New Version**
- Learning system with pattern matching
- Accuracy statistics tracking
- Trigger bot functionality
- Target trajectory prediction
- Player profile system
- Overshoot detection and correction

### **6. Key Features in New Version**
- Anti-shake system to prevent micro-jitter
- Professional aiming enhancement (donk-inspired)
- Mouse path recording/playback
- Faster response times (~15ms target)
- Smoother transitions between body parts

## Architecture Diagram:

```mermaid
graph TB
    subgraph "New Aimbot Architecture"
        A[Aimbot Main Class] --> B[OptimizedAimingController]
        A --> C[ProAimEnhancer]
        B --> D[MousePathRecorder]
        
        B --> E[Target Tracking]
        B --> F[Body Part System]
        
        C --> G[Anti-Shake System]
        C --> H[Professional Smoothing]
        
        D --> I[Path Recording]
        D --> J[Path Playback]
        D --> K[Path Optimization]
        
        E --> L[Target Data Store]
        F --> M[Body Part Weights]
        
        style A fill:#4CAF50
        style B fill:#2196F3
        style C fill:#FF9800
        style D fill:#9C27B0
    end
    
    subgraph "Input/Output Flow"
        N[Screen Capture] --> O[YOLO Detection]
        O --> P[Target Selection]
        P --> Q[Aim Point Calculation]
        Q --> R[Movement Generation]
        R --> S[Mouse Movement]
        S --> T[Path Recording]
        T --> R
    end
```


## Migration Path (Old → New):
The new version simplifies the architecture but loses some useless features.

1. **No stats accuracy** the accuracy statistics tracking
2. **No integrate that does not works** the learning system for parameter adjustment
3. **What I added** the professional enhancement and anti-shake systems
4. **Maintain** the new mouse path recording system

The new architecture is cleaner and faster, but the old version had more sophisticated learning capabilities that almost looks useless.
