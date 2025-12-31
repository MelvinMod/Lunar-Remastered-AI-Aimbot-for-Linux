import ctypes
import cv2
import json
import math
import mss
import os
import sys
import time
import torch
import numpy as np
import random
from termcolor import colored
from ultralytics import YOLO
import platform
import subprocess
from collections import deque
import threading

# ============================================================================
# INPUT STRUCTURES
# ============================================================================

class PUL(ctypes.POINTER(ctypes.c_ulong)):
    pass

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

if platform.system() == 'Windows':
    screensize = {'X': ctypes.windll.user32.GetSystemMetrics(0), 
                  'Y': ctypes.windll.user32.GetSystemMetrics(1)}
else:
    if PYAUTOGUI_AVAILABLE:
        screensize = {'X': pyautogui.size().width, 
                      'Y': pyautogui.size().height}
    else:
        try:
            result = subprocess.run(['xrandr'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if '*' in line:
                    resolution = line.split()[0]
                    w, h = map(int, resolution.split('x'))
                    screensize = {'X': w, 'Y': h}
                    break
        except:
            screensize = {'X': 1920, 'Y': 1080}

screen_res_x = screensize['X']
screen_res_y = screensize['Y']
screen_x = int(screen_res_x / 2)
screen_y = int(screen_res_y / 2)

# ============================================================================
# MOUSE PATH RECORDER & PLAYBACK SYSTEM
# ============================================================================

class MousePathRecorder:
    """Records mouse movements and creates smooth paths"""
    def __init__(self):
        self.recording = False
        self.mouse_paths = []
        self.current_path = []
        self.last_position = None
        self.last_time = None
        
        # Try to load existing recorded paths
        self.load_paths()
    
    def start_recording(self):
        """Start recording mouse movements"""
        self.recording = True
        self.current_path = []
        print("[RECORDING] Started recording mouse paths")
    
    def stop_recording(self):
        """Stop recording"""
        self.recording = False
        if self.current_path:
            self.mouse_paths.append(self.current_path.copy())
            self.save_paths()
            print(f"[RECORDING] Saved path with {len(self.current_path)} movements")
    
    def record_movement(self, dx, dy):
        """Record a mouse movement"""
        if not self.recording:
            return
        
        current_time = time.time()
        movement = {
            'dx': dx,
            'dy': dy,
            'timestamp': current_time,
            'speed': math.sqrt(dx*dx + dy*dy) / max(0.001, current_time - self.last_time if self.last_time else 0.001)
        }
        
        self.current_path.append(movement)
        self.last_time = current_time
        
        # Limit path length
        if len(self.current_path) > 1000:
            self.current_path = self.current_path[-500:]
    
    def get_optimized_path(self, required_distance, target_direction):
        """Get an optimized path based on required distance and direction"""
        if not self.mouse_paths:
            return None
        
        # Filter paths by similarity to required distance
        suitable_paths = []
        for path in self.mouse_paths:
            if len(path) < 3:
                continue
            
            # Calculate total distance of path
            path_distance = sum(math.sqrt(m['dx']**2 + m['dy']**2) for m in path)
            
            # Check if path distance is within 30% of required distance
            if abs(path_distance - required_distance) < required_distance * 0.3:
                suitable_paths.append((path, path_distance))
        
        if not suitable_paths:
            return None
        
        # Choose the best matching path
        best_path = min(suitable_paths, key=lambda x: abs(x[1] - required_distance))[0]
        
        # Extract just the movement vectors
        movements = [(m['dx'], m['dy']) for m in best_path]
        
        # Adjust direction to match target
        adjusted_movements = self.adjust_direction(movements, target_direction)
        
        # Adjust speed based on required distance
        final_movements = self.adjust_speed(adjusted_movements, required_distance)
        
        return final_movements
    
    def adjust_direction(self, movements, target_direction):
        """Adjust movements to point toward target direction"""
        if not movements:
            return movements
        
        # Calculate current direction of path
        path_dx = sum(dx for dx, dy in movements)
        path_dy = sum(dy for dx, dy in movements)
        path_direction = math.atan2(path_dy, path_dx) if path_dx != 0 or path_dy != 0 else 0
        
        # Calculate rotation needed
        rotation_angle = target_direction - path_direction
        
        # Rotate all movements
        rotated_movements = []
        for dx, dy in movements:
            distance = math.sqrt(dx*dx + dy*dy)
            current_angle = math.atan2(dy, dx)
            new_angle = current_angle + rotation_angle
            
            new_dx = distance * math.cos(new_angle)
            new_dy = distance * math.sin(new_angle)
            
            rotated_movements.append((new_dx, new_dy))
        
        return rotated_movements
    
    def adjust_speed(self, movements, required_distance):
        """Adjust movement speed to match required distance"""
        if not movements:
            return movements
        
        # Calculate current total distance
        current_distance = sum(math.sqrt(dx*dx + dy*dy) for dx, dy in movements)
        
        if current_distance == 0:
            return movements
        
        # Calculate scaling factor
        scale_factor = required_distance / current_distance
        
        # Apply scaling
        scaled_movements = [(dx * scale_factor, dy * scale_factor) for dx, dy in movements]
        
        return scaled_movements
    
    def save_paths(self):
        """Save recorded paths to file"""
        try:
            with open("mouse_paths.json", 'w') as f:
                # Convert to serializable format
                serializable_paths = []
                for path in self.mouse_paths[-10:]:  # Keep last 10 paths
                    serializable_path = []
                    for m in path:
                        serializable_path.append({
                            'dx': float(m['dx']),
                            'dy': float(m['dy']),
                            'speed': float(m.get('speed', 0))
                        })
                    serializable_paths.append(serializable_path)
                
                json.dump(serializable_paths, f, indent=2)
            print(f"[RECORDING] Saved {len(self.mouse_paths)} paths")
        except:
            pass
    
    def load_paths(self):
        """Load recorded paths from file"""
        try:
            if os.path.exists("mouse_paths.json"):
                with open("mouse_paths.json", 'r') as f:
                    loaded_paths = json.load(f)
                
                self.mouse_paths = []
                for path in loaded_paths:
                    self.mouse_paths.append(path)
                
                print(f"[RECORDING] Loaded {len(self.mouse_paths)} recorded paths")
                return True
        except:
            pass
        return False

# ============================================================================
# OPTIMIZED SMOOTH AIMING CONTROLLER
# ============================================================================

class OptimizedAimingController:
    def __init__(self, box_constant=320):
        self.box_constant = box_constant
        self.path_recorder = MousePathRecorder()
        
        # Performance stats
        self.performance_stats = {
            'avg_response_time': 0.015,
            'move_count': 0
        }
        
        # OPTIMIZED SPEED SETTINGS (FAST)
        self.base_speed = 4.2          # Increased speed
        self.smoothing = 0.75          # Smooth but fast
        self.max_step_size = 45        # Larger steps for speed
        self.deadzone_radius = 4       # Small deadzone
        self.min_move_threshold = 0.5
        
        # Body part definitions
        self.body_parts = {
            'head': {'ratio': 0.25, 'weight': 1.0},
            'neck': {'ratio': 0.33, 'weight': 0.9},
            'chest': {'ratio': 0.40, 'weight': 0.8},
            'belly': {'ratio': 0.55, 'weight': 0.7},
            'groin': {'ratio': 0.70, 'weight': 0.6},
            'legs': {'ratio': 0.85, 'weight': 0.5}
        }
        
        # Target tracking
        self.target_data = {}
        
        # Start recording immediately (ALWAYS RECORDING)
        self.start_recording()
        
        print("[CONTROLLER] Optimized aiming controller initialized")
        print(f"[SPEED] Base speed: {self.base_speed}, Smoothing: {self.smoothing}")
    
    def start_recording(self):
        """Start recording mouse movements"""
        self.path_recorder.start_recording()
    
    def stop_recording(self):
        """Stop recording"""
        self.path_recorder.stop_recording()
    
    def record_movement(self, dx, dy):
        """Record a mouse movement"""
        self.path_recorder.record_movement(dx, dy)
    
    def get_crosshair_position(self):
        return self.box_constant // 2, self.box_constant // 2
    
    def get_closest_body_part(self, target_bbox):
        x1, y1, x2, y2 = target_bbox
        width = x2 - x1
        height = y2 - y1
        cross_x, cross_y = self.get_crosshair_position()
        
        part_distances = {}
        for part, data in self.body_parts.items():
            part_x = x1 + (width // 2)
            part_y = y1 + (height * data['ratio'])
            dist = math.sqrt((part_x - cross_x)**2 + (part_y - cross_y)**2)
            weighted_dist = dist / data['weight']
            part_distances[part] = weighted_dist
        
        return min(part_distances, key=part_distances.get)
    
    def get_smooth_aim_point(self, target_bbox, target_id):
        x1, y1, x2, y2 = target_bbox
        width = x2 - x1
        height = y2 - y1
        current_time = time.time()
        
        closest_part = self.get_closest_body_part(target_bbox)
        
        if target_id not in self.target_data:
            upper_options = ['chest', 'neck', 'head']
            target_part = random.choice(upper_options)
            
            self.target_data[target_id] = {
                'current_part': closest_part,
                'target_part': target_part,
                'transition_progress': 0.0,
                'last_seen': current_time,
                'last_closest': closest_part
            }
        else:
            target_info = self.target_data[target_id]
            target_info['last_seen'] = current_time
            
            if closest_part != target_info['last_closest']:
                target_info['last_closest'] = closest_part
                target_info['current_part'] = closest_part
                target_info['transition_progress'] = 0.0
                upper_options = ['chest', 'neck', 'head']
                target_info['target_part'] = random.choice(upper_options)
            
            if target_info['current_part'] != target_info['target_part']:
                current_ratio = self.body_parts[target_info['current_part']]['ratio']
                target_ratio = self.body_parts[target_info['target_part']]['ratio']
                
                transition_speed = 0.04
                target_info['transition_progress'] = min(
                    1.0, 
                    target_info['transition_progress'] + transition_speed
                )
                
                if target_info['transition_progress'] >= 1.0:
                    target_info['current_part'] = target_info['target_part']
                    target_info['transition_progress'] = 0.0
        
        target_info = self.target_data[target_id]
        
        if target_info['transition_progress'] > 0:
            start_ratio = self.body_parts[target_info['current_part']]['ratio']
            end_ratio = self.body_parts[target_info['target_part']]['ratio']
            progress = target_info['transition_progress']
            ease_progress = 1 - (1 - progress) ** 2  # Faster easing
            current_ratio = start_ratio + (end_ratio - start_ratio) * ease_progress
        else:
            current_ratio = self.body_parts[target_info['current_part']]['ratio']
        
        aim_x = x1 + (width // 2)
        aim_y = y1 + (height * current_ratio)
        
        offset_x = random.uniform(-width * 0.015, width * 0.015)
        offset_y = random.uniform(-height * 0.01, height * 0.01)
        
        return aim_x + offset_x, aim_y + offset_y
    
    def calculate_movement_from_recorded_path(self, dx, dy):
        """Use recorded mouse paths for movement"""
        required_distance = math.sqrt(dx*dx + dy*dy)
        target_direction = math.atan2(dy, dx)
        
        # Try to get a recorded path
        recorded_movements = self.path_recorder.get_optimized_path(required_distance, target_direction)
        
        if recorded_movements:
            # Use recorded path
            total_dx = sum(move[0] for move in recorded_movements)
            total_dy = sum(move[1] for move in recorded_movements)
            
            # Scale to match exact required distance
            recorded_distance = math.sqrt(total_dx**2 + total_dy**2)
            if recorded_distance > 0:
                scale = required_distance / recorded_distance
                total_dx *= scale
                total_dy *= scale
        else:
            # Use optimized calculation
            speed_mult = self.base_speed
            smooth = self.smoothing
            
            if required_distance > 80:
                speed_mult *= 1.3
            elif required_distance < 15:
                speed_mult *= 0.9
            
            total_dx = dx * speed_mult * smooth
            total_dy = dy * speed_mult * smooth
        
        return total_dx, total_dy
    
    def execute_optimized_movement(self, dx, dy, target_speed=0, distance=0):
        """Execute fast optimized movement"""
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return
        
        start_time = time.perf_counter()
        
        # Calculate movement using recorded paths
        move_dx, move_dy = self.calculate_movement_from_recorded_path(dx, dy)
        
        # Apply speed boost for moving targets
        if target_speed > 20:
            move_dx *= 1.25
            move_dy *= 1.25
        
        # Record this movement for future learning
        self.path_recorder.record_movement(move_dx, move_dy)
        
        # Limit step size
        move_distance = math.sqrt(move_dx**2 + move_dy**2)
        if move_distance > self.max_step_size:
            scale = self.max_step_size / move_distance
            move_dx *= scale
            move_dy *= scale
        
        # Convert to integers
        move_dx_int = int(round(move_dx))
        move_dy_int = int(round(move_dy))
        
        if move_dx_int == 0 and move_dy_int == 0:
            return
        
        # FAST execution (no delay)
        try:
            subprocess.run(
                ["xdotool", "mousemove_relative", "--",
                 str(move_dx_int), str(move_dy_int)],
                capture_output=True,
                timeout=0.01
            )
        except:
            pass
        
        # Update stats
        response_time = time.perf_counter() - start_time
        self.performance_stats['avg_response_time'] = (
            0.95 * self.performance_stats['avg_response_time'] + 0.05 * response_time
        )
        self.performance_stats['move_count'] += 1
    
    def get_aim_point(self, target_bbox, target_id):
        return self.get_smooth_aim_point(target_bbox, target_id)

# ============================================================================
# PROFESSIONAL AIM STYLE (donk-inspired but keeps YOUR style)
# ============================================================================

class ProAimEnhancer:
    """Enhances your aim with professional techniques while keeping your style"""
    def __init__(self):
        # Keep your base settings but enhance them
        self.consistency_boost = 1.1  # Makes your aim more consistent
        self.headshot_focus = 0.7     # 70% chance for headshots (not 85% to keep your style)
        self.smooth_lock = 0.9        # Smoother lock when close
        
        # Anti-shake system
        self.last_movement = (0, 0)
        self.movement_buffer = deque(maxlen=3)
        self.shake_threshold = 1.5
        
    def enhance_movement(self, dx, dy, target_bbox=None):
        """Enhance your movement with professional smoothness"""
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Anti-shake: Don't make tiny back-and-forth movements
        if len(self.movement_buffer) >= 2:
            avg_dx = np.mean([m[0] for m in self.movement_buffer])
            avg_dy = np.mean([m[1] for m in self.movement_buffer])
            
            # Check if we're shaking (changing direction rapidly)
            current_dir = math.atan2(dy, dx)
            avg_dir = math.atan2(avg_dy, avg_dx)
            dir_diff = abs(((current_dir - avg_dir + math.pi) % (2 * math.pi)) - math.pi)
            
            if dir_diff > math.radians(60) and distance < 10:
                # Likely shaking - use average direction
                dx = avg_dx * 0.7 + dx * 0.3
                dy = avg_dy * 0.7 + dy * 0.3
                distance = math.sqrt(dx*dx + dy*dy)
        
        # Store in buffer
        self.movement_buffer.append((dx, dy))
        
        # Apply professional smoothness
        if distance < 5:
            # Very close - extra smooth
            smooth_factor = 0.9
        elif distance < 20:
            # Medium distance - balanced
            smooth_factor = 0.8
        else:
            # Far distance - your original style
            smooth_factor = 0.75
        
        # Apply consistency boost
        enhanced_dx = dx * self.consistency_boost * smooth_factor
        enhanced_dy = dy * self.consistency_boost * smooth_factor
        
        return enhanced_dx, enhanced_dy
    
    def enhance_aim_point(self, aim_x, aim_y, target_bbox, is_headshot=False):
        """Enhance aim point with professional precision"""
        x1, y1, x2, y2 = target_bbox
        width = x2 - x1
        height = y2 - y1
        
        if is_headshot and random.random() < self.headshot_focus:
            # Professional headshot precision
            aim_y = y1 + height * 0.22  # Slightly higher for headshots
            # Reduce random offset for headshots
            offset_scale = 0.5
        else:
            # Your original style
            offset_scale = 1.0
        
        # Add small professional offset (less random, more deliberate)
        offset_x = random.uniform(-width * 0.01, width * 0.01) * offset_scale
        offset_y = random.uniform(-height * 0.008, height * 0.008) * offset_scale
        
        return aim_x + offset_x, aim_y + offset_y

# ============================================================================
# MAIN AIMBOT CLASS (FIXED - No jerking, keeps your style)
# ============================================================================

class Aimbot:
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    screen = mss.mss()
    aimbot_status = colored("ENABLED", 'green')
    
    def __init__(self, box_constant=400, collect_data=False):
        self.box_constant = box_constant
        
        print(colored("\n[+] Loading AI Model...", "cyan"))
        try:
            self.model = YOLO('lib/best.pt')
            if torch.cuda.is_available():
                print(colored("[+] CUDA Acceleration: ENABLED", "green"))
            else:
                print(colored("[!] CUDA: Not available (CPU mode)", "yellow"))
        except Exception as e:
            print(colored(f"[!] Model Error: {e}", "red"))
            self.model = None
        
        self.confidence = 0.35
        self.iou = 0.35
        
        # Your original controller
        self.aim_controller = OptimizedAimingController(box_constant=self.box_constant)
        
        # Professional enhancer (doesn't replace, just enhances)
        self.pro_enhancer = ProAimEnhancer()
        
        self.target_fps = 240
        self.running = True
        
        print(colored("\n[+] Professional Enhanced Aimbot Initialized", "green"))
        print(colored(f"    • Target FPS: {self.target_fps}", "cyan"))
        print(colored(f"    • Your Speed: {self.aim_controller.base_speed}", "cyan"))
        print(colored(f"    • Your Smoothing: {self.aim_controller.smoothing}", "cyan"))
        print(colored("    • Anti-shake system: ENABLED", "cyan"))
        print(colored("    • Professional enhancement: ENABLED", "cyan"))
        print(colored("    • Your style + Pro precision = Perfect aim", "cyan"))
        print(colored("\n[CONTROLS]", "yellow"))
        print(colored("    • Mouse Button 4: Toggle Aimbot", "white"))
        print(colored("    • Mouse Button 5: Exit", "white"))
        print(colored("\n[NOTE] ESC and Q keys disabled - use Mouse Button 5 to exit", "yellow"))
    
    @staticmethod
    def is_aimbot_enabled():
        return Aimbot.aimbot_status == colored("ENABLED", 'green')
    
    @staticmethod
    def is_target_locked(x, y):
        # Your original threshold
        threshold = 6
        return (screen_x - threshold <= x <= screen_x + threshold and 
                screen_y - threshold <= y <= screen_y + threshold)
    
    def get_aim_point(self, target_bbox, target_id):
        # Get your original aim point
        aim_point = self.aim_controller.get_aim_point(target_bbox, target_id)
        
        if isinstance(aim_point, tuple) and len(aim_point) >= 2:
            aim_x, aim_y = aim_point
            # Enhance with professional precision
            enhanced_point = self.pro_enhancer.enhance_aim_point(
                aim_x, aim_y, target_bbox, 
                is_headshot=random.random() < 0.7  # 70% headshot focus
            )
            return enhanced_point
        
        # Fallback to center
        x1, y1, x2, y2 = target_bbox
        return (x1 + x2) // 2, (y1 + y2) // 2
    
    def execute_smooth_aim(self, dx, dy, target_speed=0, distance=0, target_bbox=None):
        """Execute aim with anti-shake and professional smoothness"""
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return
        
        # Enhance movement with professional smoothness
        enhanced_dx, enhanced_dy = self.pro_enhancer.enhance_movement(dx, dy, target_bbox)
        
        # Use your original controller but with enhanced movement
        # We'll temporarily replace the controller's calculation
        
        # Calculate required distance and direction
        required_distance = math.sqrt(enhanced_dx*enhanced_dx + enhanced_dy*enhanced_dy)
        target_direction = math.atan2(enhanced_dy, enhanced_dx)
        
        # Try to get a recorded path from YOUR style
        recorded_movements = self.aim_controller.path_recorder.get_optimized_path(
            required_distance, target_direction
        )
        
        if recorded_movements:
            # Use YOUR recorded path
            total_dx = sum(move[0] for move in recorded_movements)
            total_dy = sum(move[1] for move in recorded_movements)
            
            # Scale to match exact required distance
            recorded_distance = math.sqrt(total_dx**2 + total_dy**2)
            if recorded_distance > 0:
                scale = required_distance / recorded_distance
                total_dx *= scale
                total_dy *= scale
        else:
            # Use YOUR optimized calculation
            speed_mult = self.aim_controller.base_speed
            smooth = self.aim_controller.smoothing
            
            if required_distance > 80:
                speed_mult *= 1.3
            elif required_distance < 15:
                speed_mult *= 0.9
            
            total_dx = enhanced_dx * speed_mult * smooth
            total_dy = enhanced_dy * speed_mult * smooth
        
        # Apply speed boost for moving targets
        if target_speed > 20:
            total_dx *= 1.25
            total_dy *= 1.25
        
        # Record this movement for YOUR future learning
        self.aim_controller.path_recorder.record_movement(total_dx, total_dy)
        
        # Limit step size
        move_distance = math.sqrt(total_dx**2 + total_dy**2)
        if move_distance > self.aim_controller.max_step_size:
            scale = self.aim_controller.max_step_size / move_distance
            total_dx *= scale
            total_dy *= scale
        
        # Convert to integers
        move_dx_int = int(round(total_dx))
        move_dy_int = int(round(total_dy))
        
        if move_dx_int == 0 and move_dy_int == 0:
            return
        
        # Execute
        try:
            subprocess.run(
                ["xdotool", "mousemove_relative", "--",
                 str(move_dx_int), str(move_dy_int)],
                capture_output=True,
                timeout=0.01
            )
        except:
            pass
    
    def ultra_fast_aim(self, target_info, target_id):
        if not Aimbot.is_aimbot_enabled():
            return False
        
        start_time = time.perf_counter()
        
        x1, y1 = target_info["x1y1"]
        x2, y2 = target_info["x2y2"]
        distance = target_info["distance"]
        
        aim_x, aim_y = self.get_aim_point((x1, y1, x2, y2), target_id)
        
        absolute_x = aim_x + target_info.get('box_left', 0)
        absolute_y = aim_y + target_info.get('box_top', 0)
        
        dx = absolute_x - screen_x
        dy = absolute_y - screen_y
        
        # Don't move if very close (prevents shaking)
        if abs(dx) < 1.5 and abs(dy) < 1.5:
            return True
        
        # Calculate target speed
        target_speed = 0
        
        # Execute smooth aim
        self.execute_smooth_aim(dx, dy, target_speed, distance, (x1, y1, x2, y2))
        
        locked = Aimbot.is_target_locked(absolute_x, absolute_y)
        
        response_time = (time.perf_counter() - start_time) * 1000
        if response_time > 20:
            print(f"[RESPONSE] {response_time:.1f}ms")
        
        return locked
    
    def start(self, is_aimbot_enabled_fn=None):
        print(colored("\n[+] Starting capture...", "green"))
        
        box_size = 320
        detection_box = {
            'left': int(screen_res_x // 2 - box_size // 2),
            'top': int(screen_res_y // 2 - box_size // 2),
            'width': box_size,
            'height': box_size
        }
        
        frame_count = 0
        last_fps_time = time.time()
        fps = 0
        
        try:
            while self.running:
                frame_start = time.perf_counter()
                
                aimbot_enabled = True
                if is_aimbot_enabled_fn:
                    aimbot_enabled = is_aimbot_enabled_fn()
                
                screenshot = self.screen.grab(detection_box)
                frame = np.array(screenshot, dtype=np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                closest_target = None
                closest_distance = float('inf')
                closest_id = None
                
                if self.model:
                    try:
                        results = self.model.predict(
                            source=frame,
                            conf=self.confidence,
                            iou=self.iou,
                            verbose=False,
                            half=True,
                            imgsz=320,
                            max_det=3
                        )
                        
                        if results and len(results) > 0:
                            boxes = results[0].boxes
                            if boxes is not None and len(boxes) > 0:
                                for i, box in enumerate(boxes.xyxy):
                                    x1, y1, x2, y2 = map(int, box[:4])
                                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                                    
                                    dist = math.dist((center_x, center_y), 
                                                   (box_size // 2, box_size // 2))
                                    
                                    if dist < closest_distance:
                                        closest_distance = dist
                                        closest_id = f"target_{i}_{time.time():.3f}"
                                        
                                        closest_target = {
                                            "x1y1": (x1, y1),
                                            "x2y2": (x2, y2),
                                            "height": y2 - y1,
                                            "center": (center_x, center_y),
                                            "distance": dist,
                                            "box_left": detection_box['left'],
                                            "box_top": detection_box['top']
                                        }
                    except:
                        pass
                
                if closest_target and aimbot_enabled:
                    locked = self.ultra_fast_aim(closest_target, closest_id)
                    
                    status_color = (0, 255, 0) if locked else (0, 165, 255)
                    status_text = "LOCKED" if locked else "TRACKING"
                    
                    cv2.putText(frame, status_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_fps_time = current_time
                
                cv2.putText(frame, f"FPS: {fps}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
                
                status_color = (0, 255, 0) if aimbot_enabled else (0, 0, 255)
                status_text = "AIMBOT: ON" if aimbot_enabled else "AIMBOT: OFF"
                cv2.putText(frame, status_text, (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Display style info
                cv2.putText(frame, f"Your Style + Pro Enhance", (10, 120),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 2)
                
                cv2.putText(frame, "Anti-Shake: ACTIVE", (10, 140),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 200), 2)
                
                cv2.imshow("Professional Enhanced Aimbot", frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty("Professional Enhanced Aimbot", cv2.WND_PROP_VISIBLE) < 1:
                    break
                
                loop_time = time.perf_counter() - frame_start
                target_time = 1.0 / self.target_fps
                
                if loop_time < target_time:
                    time.sleep(target_time - loop_time)
        
        except KeyboardInterrupt:
            print(colored("\n[!] Interrupted", "yellow"))
        finally:
            cv2.destroyAllWindows()
            self.aim_controller.stop_recording()
            
            print(colored("\n" + "="*50, "cyan"))
            print(colored("PERFORMANCE STATISTICS", "cyan", attrs=['bold']))
            print(colored("="*50, "cyan"))
            stats = self.aim_controller.performance_stats
            print(f"Average Response Time: {stats['avg_response_time']*1000:.1f}ms")
            print(f"Total Movements: {stats['move_count']}")
            print(colored("="*50, "cyan"))
