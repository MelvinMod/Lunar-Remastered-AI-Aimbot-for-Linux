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
import statistics

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
# REAL-TIME MOUSE PATH MANAGER
# ============================================================================

class RealTimeMousePathManager:
    """Manages mouse paths in real-time with editing and optimization"""
    def __init__(self):
        self.mouse_paths = []  # List of dictionaries with path data
        self.current_path = []
        self.path_history = deque(maxlen=30)  # Track which paths are used
        
        # Learning parameters
        self.max_paths = 6
        self.min_path_length = 3
        self.max_path_age = 30  # seconds
        self.edit_probability = 0.02  # 2% chance to edit a path
        self.keep_probability = 0.85  # 85% chance to keep a good path
        
        # Anti-shake parameters
        self.min_movement_threshold = 0.3
        self.max_direction_change = math.radians(25)
        
        # Load existing paths
        self.load_paths()
        
        print(f"[PATH MANAGER] Initialized with {len(self.mouse_paths)} paths")
    
    def add_movement(self, dx, dy, is_aimbot_movement=False):
        """Add movement to current path with anti-shake"""
        current_time = time.time()
        
        # Filter out micro-shakes
        movement_mag = math.sqrt(dx*dx + dy*dy)
        if movement_mag < self.min_movement_threshold:
            return
        
        # Calculate direction
        direction = math.atan2(dy, dx)
        
        # Prevent crazy direction changes by smoothing
        if self.current_path and len(self.current_path) > 0:
            last_dx, last_dy, last_time = self.current_path[-1]
            last_direction = math.atan2(last_dy, last_dx)
            direction_diff = abs(((direction - last_direction + math.pi) % (2 * math.pi)) - math.pi)
            
            if direction_diff > self.max_direction_change:
                # Smooth the direction change
                direction = last_direction + np.sign(((direction - last_direction + math.pi) % (2 * math.pi)) - math.pi) * self.max_direction_change
        
        self.current_path.append((dx, dy, current_time))
        
        # Auto-save path if it's long enough and there's a pause
        if len(self.current_path) >= 15 and (current_time - self.current_path[-1][2] > 0.1):
            self.save_current_path()
    
    def save_current_path(self):
        """Save current path if it meets criteria"""
        if len(self.current_path) < self.min_path_length:
            self.current_path = []
            return
        
        # Calculate path characteristics
        total_dx = sum(p[0] for p in self.current_path)
        total_dy = sum(p[1] for p in self.current_path)
        total_distance = math.sqrt(total_dx*total_dx + total_dy*total_dy)
        
        # Only save meaningful paths
        if total_distance > 5:  # Minimum meaningful distance
            path_data = {
                'movements': self.current_path.copy(),
                'created': time.time(),
                'last_used': time.time(),
                'usage_count': 0,
                'total_distance': total_distance,
                'avg_speed': total_distance / max(0.001, self.current_path[-1][2] - self.current_path[0][2]),
                'direction': math.atan2(total_dy, total_dx)
            }
            
            self.mouse_paths.append(path_data)
            self.path_history.append(len(self.mouse_paths) - 1)
            
            print(f"[PATH] Saved new path with {len(self.current_path)} movements")
        
        self.current_path = []
    
    def get_random_path(self, required_distance, target_direction):
        """Get a random path that matches requirements"""
        if not self.mouse_paths:
            return None
        
        current_time = time.time()
        
        # Clean up old paths first
        self.mouse_paths = [p for p in self.mouse_paths 
                           if current_time - p['created'] < self.max_path_age]
        
        if not self.mouse_paths:
            return None
        
        # Score paths based on similarity
        scored_paths = []
        for i, path in enumerate(self.mouse_paths):
            distance_diff = abs(path['total_distance'] - required_distance) / max(required_distance, 1)
            direction_diff = abs(((path['direction'] - target_direction + math.pi) % (2 * math.pi)) - math.pi) / math.pi
            
            # Calculate score (lower is better)
            score = distance_diff * 0.6 + direction_diff * 0.4
            scored_paths.append((score, i, path))
        
        # Sort by score and pick from top 3
        scored_paths.sort(key=lambda x: x[0])
        top_paths = scored_paths[:min(3, len(scored_paths))]
        
        if not top_paths:
            return None
        
        # Randomly choose from top paths
        chosen_score, chosen_idx, chosen_path = random.choice(top_paths)
        
        # Update usage stats
        chosen_path['last_used'] = time.time()
        chosen_path['usage_count'] += 1
        
        # Occasionally edit the path
        if random.random() < self.edit_probability:
            chosen_path = self.edit_path(chosen_path)
            self.mouse_paths[chosen_idx] = chosen_path
        
        return chosen_path['movements']
    
    def edit_path(self, path_data):
        """Edit a path to improve it"""
        movements = path_data['movements']
        
        if len(movements) < 3:
            return path_data
        
        edit_type = random.choice(['speed', 'rotate', 'invert', 'smooth'])
        
        if edit_type == 'speed':
            # Change speed by random factor (0.8 to 1.2)
            factor = random.uniform(0.8, 1.2)
            new_movements = [(dx * factor, dy * factor, t) for dx, dy, t in movements]
            
        elif edit_type == 'rotate':
            # Rotate by random angle (-30 to 30 degrees)
            angle = random.uniform(-math.pi/6, math.pi/6)
            new_movements = []
            for dx, dy, t in movements:
                new_dx = dx * math.cos(angle) - dy * math.sin(angle)
                new_dy = dx * math.sin(angle) + dy * math.cos(angle)
                new_movements.append((new_dx, new_dy, t))
                
        elif edit_type == 'invert':
            # Invert direction (go opposite way)
            invert_type = random.choice(['mirror', 'reverse', 'both'])
            if invert_type == 'mirror':
                new_movements = [(-dx, -dy, t) for dx, dy, t in movements]
            elif invert_type == 'reverse':
                new_movements = list(reversed(movements))
            else:  # both
                new_movements = list(reversed([(-dx, -dy, t) for dx, dy, t in movements]))
                
        else:  # smooth
            # Smooth the path by averaging adjacent movements
            new_movements = []
            for i in range(len(movements)):
                if i == 0 or i == len(movements) - 1:
                    new_movements.append(movements[i])
                else:
                    dx1, dy1, t1 = movements[i-1]
                    dx2, dy2, t2 = movements[i]
                    dx3, dy3, t3 = movements[i+1]
                    avg_dx = (dx1 + dx2 + dx3) / 3
                    avg_dy = (dy1 + dy2 + dy3) / 3
                    new_movements.append((avg_dx, avg_dy, t2))
        
        # Update path data
        total_dx = sum(p[0] for p in new_movements)
        total_dy = sum(p[1] for p in new_movements)
        path_data['movements'] = new_movements
        path_data['total_distance'] = math.sqrt(total_dx*total_dx + total_dy*total_dy)
        path_data['direction'] = math.atan2(total_dy, total_dx)
        
        print(f"[EDIT] Path edited with {edit_type} modification")
        return path_data
    
    def manage_paths(self):
        """Manage path collection - remove unused paths"""
        if len(self.mouse_paths) <= self.max_paths:
            return
        
        current_time = time.time()
        
        # Score paths based on usage and age
        scored_paths = []
        for i, path in enumerate(self.mouse_paths):
            age_score = (current_time - path['created']) / self.max_path_age
            usage_score = 1.0 / (path['usage_count'] + 1)
            last_used_score = (current_time - path['last_used']) / self.max_path_age
            
            total_score = age_score * 0.3 + usage_score * 0.3 + last_used_score * 0.4
            scored_paths.append((total_score, i))
        
        # Sort by score (higher = worse)
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        
        # Remove worst paths
        paths_to_remove = len(self.mouse_paths) - self.max_paths
        for i in range(paths_to_remove):
            if i < len(scored_paths):
                idx = scored_paths[i][1]
                del self.mouse_paths[idx]
                print(f"[MANAGE] Removed unused path")
    
    def load_paths(self):
        """Load paths from file"""
        try:
            if os.path.exists("mouse_paths_rt.json"):
                with open("mouse_paths_rt.json", 'r') as f:
                    data = json.load(f)
                    self.mouse_paths = data.get('paths', [])
                    print(f"[LOAD] Loaded {len(self.mouse_paths)} paths")
        except:
            pass
    
    def save_paths(self):
        """Save paths to file"""
        try:
            data = {
                'paths': self.mouse_paths,
                'saved_at': time.time()
            }
            with open("mouse_paths_rt.json", 'w') as f:
                json.dump(data, f, default=str)
        except:
            pass

# ============================================================================
# REALISTIC AIMING CONTROLLER
# ============================================================================

class RealisticAimingController:
    def __init__(self, box_constant=320):
        self.box_constant = box_constant
        self.path_manager = RealTimeMousePathManager()
        
        # Realistic settings (1-3% different)
        self.base_speed = 4.2
        self.smoothing = 0.76  # 1% more smoothing
        self.max_step_size = 45
        self.deadzone_radius = 3.5  # Tighter deadzone
        
        # Anti-shake system
        self.last_positions = deque(maxlen=5)
        self.shake_threshold = 0.8
        self.stable_counter = 0
        
        # Target tracking
        self.target_data = {}
        self.last_aim_time = 0
        
        # Random path selection
        self.current_path_index = -1
        self.path_transition_smooth = 0.1
        self.last_path_switch = 0
        self.min_path_switch_time = 0.3
        
        # FIX: Store last chosen aim point per target
        self.last_aim_points = {}
        self.aim_point_consistency_threshold = 2.0  # pixels
        self.min_aim_stability_time = 0.1  # seconds
        
        print("[CONTROLLER] Realistic aiming controller initialized")
        print(f"[REALISM] Settings: Speed={self.base_speed}, Smooth={self.smoothing}")
        print("[ANTI-SHAKE] Shake prevention enabled")
        print("[PATHS] Real-time path management active")
    
    def get_crosshair_position(self):
        return self.box_constant // 2, self.box_constant // 2
    
    def get_aim_point(self, target_bbox, target_id, distance_to_target):
        """Get realistic aim point with natural variance and anti-shake"""
        x1, y1, x2, y2 = target_bbox
        width = x2 - x1
        height = y2 - y1
        
        current_time = time.time()
        
        # FIX: When very close to target, use consistent aim point
        if distance_to_target < 15:  # Very close to target
            # Check if we recently aimed at this target
            if target_id in self.last_aim_points:
                last_point, last_time = self.last_aim_points[target_id]
                # If we aimed at this target recently, use the same point
                if current_time - last_time < self.min_aim_stability_time:
                    return last_point
        
        # FIX: Choose body part based on distance for more stability
        # Closer = more headshots, further = more chest shots
        if distance_to_target < 25:  # Close range
            if random.random() < 0.7:  # 70% headshots at close range
                ratio = 0.25  # Fixed head position
                # Small variance only
                ratio += random.uniform(-0.02, 0.02)
            else:  # 30% chest at close range
                ratio = 0.4
                ratio += random.uniform(-0.03, 0.03)
        else:  # Long range
            if random.random() < 0.4:  # 40% headshots at long range
                ratio = 0.25
                ratio += random.uniform(-0.03, 0.03)
            elif random.random() < 0.8:  # 40% chest at long range
                ratio = 0.4
                ratio += random.uniform(-0.04, 0.04)
            else:  # 20% other at long range
                ratio = 0.55
                ratio += random.uniform(-0.05, 0.05)
        
        aim_x = x1 + (width // 2)
        aim_y = y1 + (height * ratio)
        
        # FIX: Reduce variance when close to target
        variance_multiplier = max(0.1, min(1.0, distance_to_target / 50))
        
        # Natural human-like variance (reduced when close)
        variance_x = width * random.uniform(-0.012, 0.012) * variance_multiplier
        variance_y = height * random.uniform(-0.01, 0.01) * variance_multiplier
        
        final_aim_point = (aim_x + variance_x, aim_y + variance_y)
        
        # Store this aim point for consistency
        self.last_aim_points[target_id] = (final_aim_point, current_time)
        
        # Clean up old aim points
        old_targets = [tid for tid, (_, ttime) in self.last_aim_points.items() 
                      if current_time - ttime > 1.0]
        for tid in old_targets:
            if tid in self.last_aim_points:
                del self.last_aim_points[tid]
        
        return final_aim_point
    
    def is_shaking(self, dx, dy):
        """Detect if aim is shaking (micro-movements)"""
        movement = math.sqrt(dx*dx + dy*dy)
        
        self.last_positions.append((dx, dy))
        
        if len(self.last_positions) < 3:
            return False
        
        # Check for back-and-forth movements
        direction_changes = 0
        for i in range(1, len(self.last_positions)):
            prev_dx, prev_dy = self.last_positions[i-1]
            curr_dx, curr_dy = self.last_positions[i]
            
            prev_dir = math.atan2(prev_dy, prev_dx)
            curr_dir = math.atan2(curr_dy, curr_dx)
            
            dir_diff = abs(((curr_dir - prev_dir + math.pi) % (2 * math.pi)) - math.pi)
            if dir_diff > math.pi * 0.7:  # Large direction change
                direction_changes += 1
        
        if direction_changes >= 2 and movement < 2:
            self.stable_counter = max(0, self.stable_counter - 1)
            return self.stable_counter < -2
        else:
            self.stable_counter = min(3, self.stable_counter + 1)
            return False
    
    def get_movement_from_paths(self, dx, dy, force_new_path=False):
        """Get movement from stored paths with smooth transitions"""
        required_distance = math.sqrt(dx*dx + dy*dy)
        target_direction = math.atan2(dy, dx)
        
        current_time = time.time()
        
        # FIX: When very close to target, use direct movement for stability
        if required_distance < 10:  # Very close to target
            # Use direct movement with increased smoothing
            speed = self.base_speed * 0.5  # Slower when close
            move_dx = dx * speed * (self.smoothing + 0.2)  # Extra smoothing
            move_dy = dy * speed * (self.smoothing + 0.2)
            return move_dx, move_dy
        
        # Check if we should switch paths
        should_switch = (force_new_path or 
                        current_time - self.last_path_switch > self.min_path_switch_time or
                        self.current_path_index == -1 or
                        random.random() < 0.15)  # 15% chance to switch
        
        if should_switch:
            # Get a new random path
            path_movements = self.path_manager.get_random_path(required_distance, target_direction)
            if path_movements:
                self.current_path_index = random.randint(0, len(self.path_manager.mouse_paths) - 1)
                self.last_path_switch = current_time
            else:
                path_movements = None
        else:
            # Continue with current path
            if 0 <= self.current_path_index < len(self.path_manager.mouse_paths):
                path = self.path_manager.mouse_paths[self.current_path_index]
                path_movements = path['movements']
            else:
                path_movements = None
        
        if path_movements:
            # Use the path
            total_dx = sum(m[0] for m in path_movements)
            total_dy = sum(m[1] for m in path_movements)
            
            # Adjust to match exact requirements
            path_distance = math.sqrt(total_dx**2 + total_dy**2)
            path_direction = math.atan2(total_dy, total_dx)
            
            if path_distance > 0:
                # Scale and rotate
                scale = required_distance / path_distance
                rotation = target_direction - path_direction
                
                # Apply to each movement
                adjusted_movements = []
                for move_dx, move_dy, _ in path_movements:
                    # Rotate
                    rot_dx = move_dx * math.cos(rotation) - move_dy * math.sin(rotation)
                    rot_dy = move_dx * math.sin(rotation) + move_dy * math.cos(rotation)
                    
                    # Scale
                    adjusted_movements.append((rot_dx * scale, rot_dy * scale))
                
                # Use first movement for immediate response
                if adjusted_movements:
                    return adjusted_movements[0]
        
        # Fallback: calculate direct movement
        speed = self.base_speed * (0.9 + 0.2 * random.random())  # 10% variance
        move_dx = dx * speed * self.smoothing
        move_dy = dy * speed * self.smoothing
        
        return move_dx, move_dy
    
    def execute_realistic_movement(self, dx, dy, target_speed=0, distance=0):
        """Execute movement with realistic feel and anti-shake"""
        # FIX: Check if we're in deadzone - stop all movement
        distance_to_target = math.sqrt(dx*dx + dy*dy)
        if distance_to_target < self.deadzone_radius:
            # In deadzone, no movement at all
            return
        
        # Check for shaking
        if self.is_shaking(dx, dy):
            # Don't move if shaking detected
            print("[ANTI-SHAKE] Shake detected, holding position")
            return
        
        # Calculate movement
        move_dx, move_dy = self.get_movement_from_paths(dx, dy)
        
        # Apply realistic speed adjustments
        if target_speed > 20:
            move_dx *= 1.2
            move_dy *= 1.2
        
        # Limit step size
        move_distance = math.sqrt(move_dx**2 + move_dy**2)
        if move_distance > self.max_step_size:
            scale = self.max_step_size / move_distance
            move_dx *= scale
            move_dy *= scale
        
        # FIX: Very small movements are noise - ignore them
        if move_distance < 0.5:
            return
        
        # Convert to integers
        move_dx_int = int(round(move_dx))
        move_dy_int = int(round(move_dy))
        
        if move_dx_int == 0 and move_dy_int == 0:
            return
        
        # Record this movement for learning
        self.path_manager.add_movement(move_dx, move_dy, True)
        
        # Execute
        try:
            subprocess.run(
                ["xdotool", "mousemove_relative", "--",
                 str(move_dx_int), str(move_dy_int)],
                capture_output=True,
                timeout=0.008
            )
        except:
            pass
        
        # Manage paths periodically
        if random.random() < 0.05:  # 5% chance each movement
            self.path_manager.manage_paths()
    
    def stop_shaking_at_target(self, target_bbox):
        """Prevent shaking when perfectly aimed"""
        x1, y1, x2, y2 = target_bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        cross_x, cross_y = self.get_crosshair_position()
        distance = math.sqrt((center_x - cross_x)**2 + (center_y - cross_y)**2)
        
        # If very close to target, increase stability
        if distance < 3:
            self.stable_counter = min(5, self.stable_counter + 1)
            return True
        
        return False

# ============================================================================
# MAIN AIMBOT CLASS
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
        
        # Realistic settings
        self.confidence = 0.38  # Slightly higher for more certainty
        self.iou = 0.32
        
        # Initialize realistic controller
        self.aim_controller = RealisticAimingController(box_constant=self.box_constant)
        
        # Performance tracking
        self.target_fps = 240
        self.running = True
        self.frame_count = 0
        self.last_path_save = time.time()
        
        # FIX: Target tracking for consistency
        self.last_target_positions = {}
        self.target_stability_threshold = 5.0  # pixels
        
        print(colored("\n[+] Realistic Aimbot Initialized", "green"))
        print(colored(f"    • Target FPS: {self.target_fps}", "cyan"))
        print(colored(f"    • Base Speed: {self.aim_controller.base_speed}", "cyan"))
        print(colored("    • Real-time path learning: ENABLED", "cyan"))
        print(colored("    • Anti-shake system: ACTIVE", "cyan"))
        print(colored("    • Path editing: ENABLED (2% chance)", "cyan"))
        print(colored("\n[FEATURES]", "yellow"))
        print(colored("    • Random path selection", "white"))
        print(colored("    • Real-time path management", "white"))
        print(colored("    • No shaking at targets", "white"))
        print(colored("    • Natural human-like variance", "white"))
        print(colored("\n[CONTROLS]", "yellow"))
        print(colored("    • Mouse Button 4: Toggle Aimbot", "white"))
        print(colored("    • Mouse Button 5: Exit", "white"))
    
    @staticmethod
    def is_aimbot_enabled():
        return Aimbot.aimbot_status == colored("ENABLED", 'green')
    
    def get_aim_point(self, target_bbox, target_id, distance_to_target):
        return self.aim_controller.get_aim_point(target_bbox, target_id, distance_to_target)
    
    def realistic_aim(self, target_info, target_id):
        if not Aimbot.is_aimbot_enabled():
            return False, False
        
        x1, y1 = target_info["x1y1"]
        x2, y2 = target_info["x2y2"]
        
        # Calculate distance from crosshair to target center
        target_center_x = (x1 + x2) // 2
        target_center_y = (y1 + y2) // 2
        cross_x, cross_y = self.aim_controller.get_crosshair_position()
        distance_to_target = math.sqrt((target_center_x - cross_x)**2 + (target_center_y - cross_y)**2)
        
        # Get aim point with distance consideration
        aim_x, aim_y = self.get_aim_point((x1, y1, x2, y2), target_id, distance_to_target)
        
        absolute_x = aim_x + target_info.get('box_left', 0)
        absolute_y = aim_y + target_info.get('box_top', 0)
        
        dx = absolute_x - screen_x
        dy = absolute_y - screen_y
        
        # FIX: Apply extra smoothing when very close to target
        current_distance = math.sqrt(dx*dx + dy*dy)
        
        # If very close, use stronger smoothing
        if current_distance < 10:
            # Scale down the movement even more when close
            scale_factor = max(0.3, current_distance / 20)
            dx *= scale_factor
            dy *= scale_factor
        
        # Check if we should stop shaking
        is_stable = self.aim_controller.stop_shaking_at_target((x1, y1, x2, y2))
        
        # FIX: Stronger deadzone when very close
        if abs(dx) < 2.0 and abs(dy) < 2.0 and is_stable:
            return True, True
        
        # Execute realistic movement with distance info
        self.aim_controller.execute_realistic_movement(dx, dy, 0, current_distance)
        
        # Check if locked
        locked = (abs(dx) < 3 and abs(dy) < 3)
        
        return locked, is_stable
    
    def start(self, is_aimbot_enabled_fn=None):
        print(colored("\n[+] Starting realistic capture...", "green"))
        
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
        
        # Also record natural mouse movements
        import Xlib.display
        display = Xlib.display.Display()
        root = display.screen().root
        
        try:
            while self.running:
                frame_start = time.perf_counter()
                
                aimbot_enabled = True
                if is_aimbot_enabled_fn:
                    aimbot_enabled = is_aimbot_enabled_fn()
                
                # Capture screen
                screenshot = self.screen.grab(detection_box)
                frame = np.array(screenshot, dtype=np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Get mouse movement (even when not aiming)
                try:
                    pointer = root.query_pointer()
                    mouse_x, mouse_y = pointer.root_x, pointer.root_y
                    
                    # Record natural mouse movements
                    if hasattr(self, 'last_mouse_pos'):
                        last_x, last_y = self.last_mouse_pos
                        dx = mouse_x - last_x
                        dy = mouse_y - last_y
                        
                        # Record all mouse movements (including crazy goofy ones)
                        self.aim_controller.path_manager.add_movement(dx, dy, False)
                    
                    self.last_mouse_pos = (mouse_x, mouse_y)
                except:
                    pass
                
                # Detect targets
                targets = []
                if self.model and aimbot_enabled:
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
                                    
                                    target_id = f"target_{i}"
                                    targets.append({
                                        "id": target_id,
                                        "x1y1": (x1, y1),
                                        "x2y2": (x2, y2),
                                        "center": (center_x, center_y),
                                        "distance": dist,
                                        "box_left": detection_box['left'],
                                        "box_top": detection_box['top']
                                    })
                    except:
                        pass
                
                # Process closest target
                if targets and aimbot_enabled:
                    closest_target = min(targets, key=lambda x: x["distance"])
                    locked, stable = self.realistic_aim(closest_target, closest_target["id"])
                    
                    # Draw target info
                    color = (0, 255, 0) if locked else (0, 165, 255)
                    thickness = 2
                    
                    x1, y1 = closest_target["x1y1"]
                    x2, y2 = closest_target["x2y2"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw status
                    if locked:
                        status = "LOCKED"
                    elif stable:
                        status = "STABLE"
                    else:
                        status = "TRACKING"
                    
                    cv2.putText(frame, status, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw distance info
                    dist_text = f"Dist: {closest_target['distance']:.1f}"
                    cv2.putText(frame, dist_text, (x1, y1 - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1)
                
                # Update FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_fps_time = current_time
                    
                    # Auto-save paths every 10 seconds
                    if current_time - self.last_path_save > 10:
                        self.aim_controller.path_manager.save_paths()
                        self.last_path_save = current_time
                        print(f"[SAVE] Paths saved ({len(self.aim_controller.path_manager.mouse_paths)} active)")
                
                # Display overlay
                cv2.putText(frame, f"FPS: {fps}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
                
                status_color = (0, 255, 0) if aimbot_enabled else (0, 0, 255)
                status_text = "REALISTIC AIM: ON" if aimbot_enabled else "REALISTIC AIM: OFF"
                cv2.putText(frame, status_text, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Display path info
                active_paths = len(self.aim_controller.path_manager.mouse_paths)
                cv2.putText(frame, f"Paths: {active_paths}/6", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 2)
                
                cv2.putText(frame, f"Speed: {self.aim_controller.base_speed}", (10, 115),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 200), 2)
                
                cv2.putText(frame, "Anti-Shake: ACTIVE", (10, 140),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
                
                cv2.putText(frame, "Distance-based aiming", (10, 165),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 255), 2)
                
                cv2.imshow("Realistic Aimbot", frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty("Realistic Aimbot", cv2.WND_PROP_VISIBLE) < 1:
                    break
                
                # Maintain FPS
                loop_time = time.perf_counter() - frame_start
                target_time = 1.0 / self.target_fps
                if loop_time < target_time:
                    time.sleep(target_time - loop_time)
        
        except KeyboardInterrupt:
            print(colored("\n[!] Interrupted", "yellow"))
        finally:
            cv2.destroyAllWindows()
            
            # Save paths on exit
            self.aim_controller.path_manager.save_paths()
            
            print(colored("\n" + "="*50, "cyan"))
            print(colored("REALISTIC PERFORMANCE SUMMARY", "cyan", attrs=['bold']))
            print(colored("="*50, "cyan"))
            print(f"Active paths saved: {len(self.aim_controller.path_manager.mouse_paths)}")
            print(f"Path edits performed: Real-time")
            print(colored("="*50, "cyan"))

# ============================================================================
# ALIAS FOR BACKWARD COMPATIBILITY
# ============================================================================

aimbot = Aimbot
