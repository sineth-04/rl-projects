import pygame
import math
import numpy as np
 
# ─── Constants ────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 900, 700
FPS = 60
CAR_W, CAR_H = 24, 44          # pixels (width, length)
WHEELBASE = 28                  # distance between axles in pixels
MAX_STEER = math.radians(35)    # max steering angle
MAX_SPEED = 3.0                 # pixels/frame
FRICTION  = 0.92                # velocity decay per frame
NUM_RAYS  = 8                   # sensor rays
RAY_LEN   = 150                 # max sensor range in pixels
 
# ─── Colours ──────────────────────────────────────────────────────────────────
COL_BG         = (30,  30,  30)
COL_ASPHALT    = (45,  45,  48)
COL_BAY_IDLE   = (80,  80,  85)
COL_BAY_TARGET = (50, 220, 120)
COL_LANE       = (220, 200,  50)
COL_CAR_AGENT  = (59, 130, 246)
COL_CAR_PARKED = (90,  95, 105)
COL_SENSOR     = (248, 113, 113)
COL_HIT        = (255,  80,  80)
COL_TEXT       = (200, 210, 220)
COL_PANEL      = (20,  22,  28, 200)
 
# ─── Parking bays ─────────────────────────────────────────────────────────────
BAY_W, BAY_H = 50, 90
BAY_ROW_Y    = 560                        # top edge of bay row
NUM_BAYS     = 11
BAY_START_X  = 40
BAYS = [(BAY_START_X + i * (BAY_W + 4), BAY_ROW_Y) for i in range(NUM_BAYS)]
TARGET_IDX   = 5                          # which bay the agent must park in
OBSTACLE_IDXS = [1, 2, 4, 6, 8, 9]      # bays with parked cars
 
 
# ─── Utility helpers ──────────────────────────────────────────────────────────
def rotate_points(points, angle, origin):
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    ox, oy = origin
    return [
        (ox + cos_a*(x - ox) - sin_a*(y - oy),
         oy + sin_a*(x - ox) + cos_a*(y - oy))
        for x, y in points
    ]
 
def rect_corners(cx, cy, w, h, angle):
    hw, hh = w / 2, h / 2
    corners = [(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)]
    return rotate_points([(cx+dx, cy+dy) for dx,dy in corners], angle, (cx,cy))
 
def poly_aabb(corners):
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys)
 
def segments_from_rect(cx, cy, w, h):
    """Return 4 line segments for an axis-aligned rect (for ray casting)."""
    l, r, t, b = cx - w/2, cx + w/2, cy - h/2, cy + h/2
    return [((l,t),(r,t)), ((r,t),(r,b)), ((r,b),(l,b)), ((l,b),(l,t))]
 
def ray_segment_intersect(ox, oy, dx, dy, p1, p2):
    """Returns distance along ray to segment intersection, or None."""
    x1,y1 = p1; x2,y2 = p2
    denom = (x2-x1)*(-dy) - (y2-y1)*(-dx)
    if abs(denom) < 1e-9:
        return None
    t = ((ox-x1)*(-dy) - (oy-y1)*(-dx)) / denom
    u = ((x2-x1)*(oy-y1) - (y2-y1)*(ox-x1)) / denom
    if 0 <= t <= 1 and u >= 0:
        return u
    return None
 
def cast_ray(ox, oy, angle, segments, max_dist=RAY_LEN):
    dx, dy = math.cos(angle), math.sin(angle)
    best = max_dist
    for seg in segments:
        d = ray_segment_intersect(ox, oy, dx, dy, *seg)
        if d is not None and d < best:
            best = d
    return best
 
 
# ─── Car ──────────────────────────────────────────────────────────────────────
class Car:
    def __init__(self, x, y, angle=0.0):
        self.x, self.y = float(x), float(y)
        self.angle  = angle        # radians, 0 = pointing up (−y)
        self.speed  = 0.0
        self.steer  = 0.0          # current steering angle
        self.gear   = 1            # 1 = forward, -1 = reverse
 
    # ── Bicycle-model kinematics (Ackermann approximation) ───────────────────
    def update(self, throttle, brake, steer_input, gear):
        self.gear = gear
        # Steering (smoothly interpolate)
        target_steer = steer_input * MAX_STEER
        self.steer += (target_steer - self.steer) * 0.25
 
        # Speed
        accel = throttle * 0.18 - brake * 0.35
        self.speed = (self.speed + accel * self.gear) * FRICTION
        self.speed = max(-MAX_SPEED, min(MAX_SPEED, self.speed))
 
        # Heading change via bicycle model
        if abs(self.steer) > 0.001 and abs(self.speed) > 0.001:
            turning_radius = WHEELBASE / math.tan(self.steer)
            angular_vel = self.speed / turning_radius
            self.angle += angular_vel
 
        # Position
        self.x += math.sin(self.angle) * self.speed
        self.y -= math.cos(self.angle) * self.speed   # pygame y-axis is flipped
 
    def corners(self):
        return rect_corners(self.x, self.y, CAR_W, CAR_H, self.angle)
 
    def sense(self, world_segments):
        angles = [self.angle + math.radians(a)
                  for a in [-135,-90,-45,-20, 20, 45, 90, 135]]
        return [cast_ray(self.x, self.y, a, world_segments) for a in angles]
 
 
# ─── Environment ──────────────────────────────────────────────────────────────
class ParkingEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Parking RL — Base Environment")
        self.clock  = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 13)
        self.font_m = pygame.font.SysFont("monospace", 15, bold=True)
        self._build_world()
        self.reset()
 
    def _build_world(self):
        # Static world segments for ray casting
        segs = []
        # Screen boundary
        segs += [((0,0),(SCREEN_W,0)), ((SCREEN_W,0),(SCREEN_W,SCREEN_H)),
                 ((SCREEN_W,SCREEN_H),(0,SCREEN_H)), ((0,SCREEN_H),(0,0))]
        # Bay dividers (vertical lines)
        for bx, by in BAYS:
            segs += [((bx, by),(bx, by+BAY_H)),
                     ((bx+BAY_W, by),(bx+BAY_W, by+BAY_H))]
        # Bottom wall
        segs += [((0, BAY_ROW_Y+BAY_H),(SCREEN_W, BAY_ROW_Y+BAY_H))]
        # Parked car segments
        for idx in OBSTACLE_IDXS:
            bx, by = BAYS[idx]
            cx = bx + BAY_W/2
            cy = by + BAY_H/2
            segs += segments_from_rect(cx, cy, CAR_W+2, CAR_H+2)
        self.world_segs = segs
 
        # Target bay rect
        tbx, tby = BAYS[TARGET_IDX]
        self.target_rect = pygame.Rect(tbx, tby, BAY_W, BAY_H)
        self.target_cx = tbx + BAY_W/2
        self.target_cy = tby + BAY_H/2
        self.target_angle = math.pi    # facing down (into the bay)
 
    def reset(self):
        # Spawn agent in a random position above the bays
        sx = np.random.uniform(80, SCREEN_W - 80)
        sy = np.random.uniform(150, 430)
        sa = np.random.uniform(-0.4, 0.4)
        self.agent = Car(sx, sy, sa)
        self.steps = 0
        self.done  = False
        self.reward_last = 0.0
        self.collision = False
        return self._obs()
 
    def step(self, action):
        throttle, brake, steer, gear = action
        self.agent.update(throttle, brake, steer, gear)
        self.steps += 1
 
        obs = self._obs()
        reward, done = self._reward_done()
        self.reward_last = reward
        self.done = done
        return obs, reward, done
 
    def _obs(self):
        a = self.agent
        dx = self.target_cx - a.x
        dy = self.target_cy - a.y
        dist = math.hypot(dx, dy)
        angle_to_target = math.atan2(dx, -dy) - a.angle  # relative angle
        angle_diff = ((a.angle - self.target_angle) + math.pi) % (2*math.pi) - math.pi
        rays = a.sense(self.world_segs)
        return np.array([a.x/SCREEN_W, a.y/SCREEN_H,
                         math.sin(a.angle), math.cos(a.angle),
                         a.speed/MAX_SPEED, a.steer/MAX_STEER,
                         dist/SCREEN_W, math.sin(angle_to_target),
                         math.cos(angle_to_target), angle_diff/math.pi,
                         float(a.gear)] + [r/RAY_LEN for r in rays],
                        dtype=np.float32)
 
    def _reward_done(self):
        a = self.agent
        corners = a.corners()
 
        # Collision check (simple AABB vs parked cars + walls)
        self.collision = False
        ax1,ay1,ax2,ay2 = poly_aabb(corners)
        for idx in OBSTACLE_IDXS:
            bx, by = BAYS[idx]
            pcx, pcy = bx+BAY_W/2, by+BAY_H/2
            pr = pygame.Rect(pcx-CAR_W/2-1, pcy-CAR_H/2-1, CAR_W+2, CAR_H+2)
            if pr.colliderect(pygame.Rect(ax1,ay1,ax2-ax1,ay2-ay1)):
                self.collision = True
        out_of_bounds = (a.x < 10 or a.x > SCREEN_W-10 or
                         a.y < 10 or a.y > SCREEN_H-10)
 
        # Distance + alignment reward
        dx = self.target_cx - a.x
        dy = self.target_cy - a.y
        dist = math.hypot(dx, dy)
        angle_diff = abs(((a.angle - self.target_angle) + math.pi) % (2*math.pi) - math.pi)
        in_bay = self.target_rect.collidepoint(a.x, a.y)
 
        reward  = -0.01                          # time penalty
        reward += max(0, (200 - dist) / 200)     # closer = better
        reward -= angle_diff * 0.3               # alignment
        reward -= 5.0 if self.collision else 0
        reward -= 5.0 if out_of_bounds   else 0
        reward += 20.0 if (in_bay and dist < 15 and angle_diff < 0.2) else 0
 
        done = (self.collision or out_of_bounds or self.steps > 1500 or
                (in_bay and dist < 15 and angle_diff < 0.2))
        return reward, done
 
    # ── Rendering ─────────────────────────────────────────────────────────────
    def render(self):
        s = self.screen
        s.fill(COL_BG)
 
        # Asphalt
        pygame.draw.rect(s, COL_ASPHALT, (0, 0, SCREEN_W, SCREEN_H))
 
        # Lane line
        pygame.draw.line(s, COL_LANE, (0, BAY_ROW_Y - 8), (SCREEN_W, BAY_ROW_Y - 8), 2)
 
        # Bays
        for i, (bx, by) in enumerate(BAYS):
            if i == TARGET_IDX:
                pygame.draw.rect(s, (*COL_BAY_TARGET, 40),
                                 (bx, by, BAY_W, BAY_H))
                pygame.draw.rect(s, COL_BAY_TARGET, (bx, by, BAY_W, BAY_H), 2)
                lbl = self.font_s.render("P", True, COL_BAY_TARGET)
                s.blit(lbl, (bx + BAY_W//2 - 5, by + BAY_H//2 - 7))
            elif i in OBSTACLE_IDXS:
                cx = bx + BAY_W//2
                cy = by + BAY_H//2
                corners = rect_corners(cx, cy, CAR_W, CAR_H, 0)
                pygame.draw.polygon(s, COL_CAR_PARKED, corners)
                pygame.draw.polygon(s, (60,65,75), corners, 1)
            else:
                pygame.draw.rect(s, COL_BAY_IDLE, (bx, by, BAY_W, BAY_H), 1)
 
        # Sensor rays
        a = self.agent
        ray_angles = [a.angle + math.radians(ang)
                      for ang in [-135,-90,-45,-20, 20, 45, 90, 135]]
        rays = a.sense(self.world_segs)
        for ang, dist in zip(ray_angles, rays):
            ex = a.x + math.cos(ang) * dist
            ey = a.y + math.sin(ang) * dist
            pygame.draw.line(s, (*COL_SENSOR, 100), (a.x, a.y), (ex, ey), 1)
            col_dot = COL_HIT if dist < RAY_LEN else COL_SENSOR
            pygame.draw.circle(s, col_dot, (int(ex), int(ey)), 3)
 
        # Agent car
        col_car = COL_HIT if self.collision else COL_CAR_AGENT
        corners = a.corners()
        pygame.draw.polygon(s, col_car, corners)
        pygame.draw.polygon(s, (150,180,255), corners, 1)
        # Windshield indicator (front)
        front_mid = ((corners[0][0]+corners[1][0])/2,
                     (corners[0][1]+corners[1][1])/2)
        pygame.draw.line(s, (150,200,255), (a.x, a.y), front_mid, 2)
 
        # ── HUD ───────────────────────────────────────────────────────────────
        hud_lines = [
            f"step    {self.steps:>5}",
            f"speed   {a.speed:>+6.2f}",
            f"steer   {math.degrees(a.steer):>+6.1f}°",
            f"gear    {'FWD' if a.gear == 1 else 'REV'}",
            f"reward  {self.reward_last:>+6.2f}",
        ]
        px, py = 14, 14
        for line in hud_lines:
            surf = self.font_s.render(line, True, COL_TEXT)
            s.blit(surf, (px, py))
            py += 18
 
        # Controls reminder
        ctrl_lines = [
            "W/S  throttle / brake",
            "A/D  steer",
            "R    reverse gear",
            "SPACE reset",
        ]
        px2 = SCREEN_W - 190
        py2 = 14
        for line in ctrl_lines:
            surf = self.font_s.render(line, True, (120,130,140))
            s.blit(surf, (px2, py2))
            py2 += 18
 
        if self.done:
            msg = "PARKED!" if not self.collision else "CRASHED"
            col = COL_BAY_TARGET if not self.collision else COL_HIT
            txt = self.font_m.render(f"── {msg} ── press SPACE", True, col)
            s.blit(txt, (SCREEN_W//2 - txt.get_width()//2, SCREEN_H//2 - 12))
 
        pygame.display.flip()
 
    # ── Manual play loop ──────────────────────────────────────────────────────
    def run_manual(self):
        gear = 1
        while True:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.reset(); gear = 1
                    if event.key == pygame.K_r:
                        gear *= -1
 
            if not self.done:
                keys = pygame.key.get_pressed()
                throttle = 1.0 if keys[pygame.K_w] else 0.0
                brake     = 1.0 if keys[pygame.K_s] else 0.0
                steer     = (-1.0 if keys[pygame.K_a] else
                              1.0 if keys[pygame.K_d] else 0.0)
                self.step((throttle, brake, steer, gear))
 
            self.render()
 
 
if __name__ == "__main__":
    env = ParkingEnv()
    env.run_manual()