import pygame
import pygame.freetype
import math
import numpy as np
import os
import glob

# ─── Constants ────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1000, 700
FPS        = 60
CAR_W, CAR_H = 24, 44
WHEELBASE  = 28
MAX_STEER  = math.radians(35)
MAX_SPEED  = 3.0
FRICTION   = 0.92
NUM_RAYS   = 12
RAY_LEN    = 180

# ─── Colours ──────────────────────────────────────────────────────────────────
COL_ASPHALT    = (45,  45,  48)
COL_BAY_IDLE   = (80,  80,  85)
COL_BAY_TARGET = (50, 220, 120)
COL_LANE       = (220, 200,  50)
COL_CAR_AGENT  = (59, 130, 246)
COL_CAR_PARKED = (90,  95, 105)
COL_SENSOR     = (248, 113, 113)
COL_HIT        = (255,  80,  80)
COL_TEXT       = (200, 210, 220)
COL_BTN        = (55,  60,  75)
COL_BTN_ACTIVE = (40, 160,  90)
COL_BTN_HOVER  = (75,  80,  98)
COL_PANEL      = (20,  22,  30)
COL_MODAL_BG   = (15,  17,  24)
COL_MODAL_BOR  = (70,  75,  95)
COL_SEL        = (50, 130, 220)

# ─── Bay layout ───────────────────────────────────────────────────────────────
BAY_W, BAY_H  = 50, 90
BAY_ROW_Y     = 560
NUM_BAYS      = 13
BAY_START_X   = 30
BAYS          = [(BAY_START_X + i*(BAY_W+4), BAY_ROW_Y) for i in range(NUM_BAYS)]
TARGET_IDX    = 6
OBSTACLE_IDXS = [1, 2, 4, 5, 7, 8, 10, 11]
MODELS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ─── Curriculum stages ────────────────────────────────────────────────────────
# Each stage: (spawn_y_min, spawn_y_max, spawn_x_radius, required_success_rate)
CURRICULUM = [
    # Stage 1: exact fixed spawn — dead centre, 100px above bay, no randomness
    {"name": "Stage 1 — Straight Fwd",  "y_min": None, "y_max": None, "x_rad": 0,   "target": 0.75, "fixed_offset": 100},
    # Stage 2: pull back further, slightly wider x
    {"name": "Stage 2 — Diagonal Fwd",  "y_min": 350,  "y_max": 490,  "x_rad": 80,  "target": 0.65, "fixed_offset": None},
    # Stage 3: fixed spawn far above bay, car faces UP — must reverse down into bay
    # Far enough that a U-turn would cost too many steps to be worth it
    {"name": "Stage 3 — Straight Rev",  "y_min": None, "y_max": None, "x_rad": 0,   "target": 0.70, "fixed_offset": 220},
    # Stage 4: reverse from varied positions, wider x spread
    {"name": "Stage 4 — Diagonal Rev",  "y_min": 200,  "y_max": 400,  "x_rad": 60,  "target": 0.60, "fixed_offset": None},
    # Stage 5: full map, parallel parking challenge
    {"name": "Stage 5 — Parallel",      "y_min": 120,  "y_max": 490,  "x_rad": 400, "target": 0.55, "fixed_offset": None},
]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def rotate_points(points, angle, origin):
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    ox, oy = origin
    return [(ox + cos_a*(x-ox) - sin_a*(y-oy),
             oy + sin_a*(x-ox) + cos_a*(y-oy)) for x, y in points]

def rect_corners(cx, cy, w, h, angle):
    hw, hh = w/2, h/2
    pts = [(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)]
    return rotate_points([(cx+dx, cy+dy) for dx,dy in pts], angle, (cx,cy))

def poly_aabb(corners):
    xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys)

def segments_from_rect(cx, cy, w, h):
    l,r,t,b = cx-w/2, cx+w/2, cy-h/2, cy+h/2
    return [((l,t),(r,t)),((r,t),(r,b)),((r,b),(l,b)),((l,b),(l,t))]

def ray_segment_intersect(ox, oy, dx, dy, p1, p2):
    x1,y1 = p1; x2,y2 = p2
    denom = (x2-x1)*(-dy) - (y2-y1)*(-dx)
    if abs(denom) < 1e-9: return None
    t = ((ox-x1)*(-dy) - (oy-y1)*(-dx)) / denom
    u = ((x2-x1)*(oy-y1) - (y2-y1)*(ox-x1)) / denom
    if 0 <= t <= 1 and u >= 0: return u
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
        self.angle = angle
        self.speed = 0.0
        self.steer = 0.0
        self.gear  = 1

    def update(self, throttle, brake, steer_input, gear):
        self.gear = gear
        self.steer += (steer_input * MAX_STEER - self.steer) * 0.25
        accel = throttle * 0.18 - brake * 0.35
        self.speed = (self.speed + accel * self.gear) * FRICTION
        max_rev = MAX_SPEED * 0.55
        self.speed = max(-max_rev, min(MAX_SPEED, self.speed))
        if abs(self.steer) > 0.001 and abs(self.speed) > 0.001:
            self.angle += self.speed / (WHEELBASE / math.tan(self.steer))
        self.x +=  math.sin(self.angle) * self.speed
        self.y -= math.cos(self.angle) * self.speed

    def corners(self):
        return rect_corners(self.x, self.y, CAR_W, CAR_H, self.angle)

    def sense(self, world_segments):
        angles = [self.angle + math.radians(i*(360/NUM_RAYS))
                  for i in range(NUM_RAYS)]
        return [cast_ray(self.x, self.y, a, world_segments) for a in angles]

# ─── Button ───────────────────────────────────────────────────────────────────
class Button:
    def __init__(self, x, y, w, h, label, color=None, active_color=None):
        self.rect  = pygame.Rect(x, y, w, h)
        self.label = label
        self.color = color or COL_BTN
        self.active_color = active_color or COL_BTN_ACTIVE
        self.active  = False
        self.hovered = False

    def draw(self, surface, font):
        col = self.active_color if self.active else (COL_BTN_HOVER if self.hovered else self.color)
        pygame.draw.rect(surface, col, self.rect, border_radius=6)
        pygame.draw.rect(surface, (90,95,120), self.rect, 1, border_radius=6)
        txt = font.render(self.label, True, COL_TEXT)
        surface.blit(txt, (self.rect.centerx - txt.get_width()//2,
                           self.rect.centery - txt.get_height()//2))

    def check_hover(self, pos): self.hovered = self.rect.collidepoint(pos)
    def is_clicked(self, pos, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(pos)

# ─── File picker modal ────────────────────────────────────────────────────────
class ModelPicker:
    """A simple in-pygame file picker modal for .zip models."""
    ITEM_H   = 32
    MODAL_W  = 500
    MODAL_H  = 420
    PAD      = 16

    def __init__(self, screen, font_s, font_m):
        self.screen  = screen
        self.font_s  = font_s
        self.font_m  = font_m
        self.visible = False
        self.models  = []
        self.selected = 0
        self.scroll   = 0
        self.max_visible = 9
        self.rect = pygame.Rect(
            (SCREEN_W - self.MODAL_W)//2,
            (SCREEN_H - self.MODAL_H)//2,
            self.MODAL_W, self.MODAL_H
        )

    def open(self):
        self.models  = sorted(glob.glob(os.path.join(MODELS_DIR, "**", "*.zip"), recursive=True))
        self.selected = 0
        self.scroll   = 0
        self.visible  = True

    def close(self): self.visible = False

    def handle_event(self, event):
        """Returns chosen path string, 'cancel', or None."""
        if not self.visible: return None
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.close(); return "cancel"
            if event.key == pygame.K_RETURN and self.models:
                path = self.models[self.selected]
                self.close(); return path
            if event.key == pygame.K_UP:
                self.selected = max(0, self.selected-1)
                self._clamp_scroll()
            if event.key == pygame.K_DOWN:
                self.selected = min(len(self.models)-1, self.selected+1)
                self._clamp_scroll()

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            # Close if clicked outside
            if not self.rect.collidepoint(mx, my):
                self.close(); return "cancel"
            # Item click
            list_top = self.rect.y + 60
            for i in range(self.max_visible):
                idx = i + self.scroll
                if idx >= len(self.models): break
                item_rect = pygame.Rect(self.rect.x + self.PAD,
                                        list_top + i*self.ITEM_H,
                                        self.MODAL_W - self.PAD*2, self.ITEM_H)
                if item_rect.collidepoint(mx, my):
                    if self.selected == idx:
                        # Double-click behaviour: single click selects, enter loads
                        pass
                    self.selected = idx
            # OK button
            ok_rect = pygame.Rect(self.rect.right - 120 - self.PAD,
                                  self.rect.bottom - 50,
                                  110, 34)
            cancel_rect = pygame.Rect(self.rect.x + self.PAD,
                                      self.rect.bottom - 50,
                                      110, 34)
            if ok_rect.collidepoint(mx, my) and self.models:
                path = self.models[self.selected]
                self.close(); return path
            if cancel_rect.collidepoint(mx, my):
                self.close(); return "cancel"

        if event.type == pygame.MOUSEWHEEL:
            self.scroll = max(0, min(self.scroll - event.y,
                                     max(0, len(self.models)-self.max_visible)))
        return None

    def _clamp_scroll(self):
        if self.selected < self.scroll:
            self.scroll = self.selected
        if self.selected >= self.scroll + self.max_visible:
            self.scroll = self.selected - self.max_visible + 1

    def draw(self):
        if not self.visible: return
        # Overlay
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0,0,0,160))
        self.screen.blit(overlay, (0,0))

        # Modal background
        pygame.draw.rect(self.screen, COL_MODAL_BG, self.rect, border_radius=10)
        pygame.draw.rect(self.screen, COL_MODAL_BOR, self.rect, 2, border_radius=10)

        # Title
        title = self.font_m.render("Select a model to load", True, COL_TEXT)
        self.screen.blit(title, (self.rect.x + self.PAD, self.rect.y + self.PAD))

        subtitle = self.font_s.render(
            f"{len(self.models)} model(s) found in /models   ↑↓ navigate   Enter to load",
            True, (100,110,130))
        self.screen.blit(subtitle, (self.rect.x + self.PAD, self.rect.y + 36))

        # Divider
        pygame.draw.line(self.screen, COL_MODAL_BOR,
                         (self.rect.x, self.rect.y+55),
                         (self.rect.right, self.rect.y+55))

        list_top = self.rect.y + 60
        if not self.models:
            msg = self.font_s.render("No .zip models found in /models folder.", True, (150,80,80))
            self.screen.blit(msg, (self.rect.x + self.PAD, list_top + 10))
        else:
            for i in range(self.max_visible):
                idx = i + self.scroll
                if idx >= len(self.models): break
                name = os.path.basename(self.models[idx])
                item_rect = pygame.Rect(self.rect.x + self.PAD,
                                        list_top + i*self.ITEM_H,
                                        self.MODAL_W - self.PAD*2,
                                        self.ITEM_H - 2)
                if idx == self.selected:
                    pygame.draw.rect(self.screen, COL_SEL, item_rect, border_radius=5)
                elif i % 2 == 0:
                    pygame.draw.rect(self.screen, (25,27,35), item_rect, border_radius=5)

                col = (255,255,255) if idx == self.selected else COL_TEXT
                txt = self.font_s.render(name, True, col)
                self.screen.blit(txt, (item_rect.x+8, item_rect.y+8))

        # Divider
        pygame.draw.line(self.screen, COL_MODAL_BOR,
                         (self.rect.x, self.rect.bottom-60),
                         (self.rect.right, self.rect.bottom-60))

        # Buttons
        ok_rect = pygame.Rect(self.rect.right-120-self.PAD, self.rect.bottom-50, 110, 34)
        cancel_rect = pygame.Rect(self.rect.x+self.PAD, self.rect.bottom-50, 110, 34)
        pygame.draw.rect(self.screen, COL_BTN_ACTIVE, ok_rect, border_radius=6)
        pygame.draw.rect(self.screen, COL_BTN, cancel_rect, border_radius=6)
        ok_txt = self.font_s.render("Load", True, (255,255,255))
        ca_txt = self.font_s.render("Cancel", True, COL_TEXT)
        self.screen.blit(ok_txt, (ok_rect.centerx-ok_txt.get_width()//2,
                                  ok_rect.centery-ok_txt.get_height()//2))
        self.screen.blit(ca_txt, (cancel_rect.centerx-ca_txt.get_width()//2,
                                  cancel_rect.centery-ca_txt.get_height()//2))

        # Scrollbar
        if len(self.models) > self.max_visible:
            sb_x = self.rect.right - 10
            sb_h = self.ITEM_H * self.max_visible
            sb_y = list_top
            ratio = self.max_visible / len(self.models)
            bar_h = max(20, int(sb_h * ratio))
            bar_y = sb_y + int((sb_h - bar_h) * self.scroll /
                               max(1, len(self.models)-self.max_visible))
            pygame.draw.rect(self.screen, (60,65,80), (sb_x, sb_y, 6, sb_h), border_radius=3)
            pygame.draw.rect(self.screen, (120,130,160), (sb_x, bar_y, 6, bar_h), border_radius=3)


# ─── Environment ──────────────────────────────────────────────────────────────
class ParkingEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Parking RL — Curriculum Training")
        self.clock  = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 13)
        self.font_m = pygame.font.SysFont("monospace", 15, bold=True)

        self.agent_mode  = "human"
        self.rl_agent    = None
        self.loaded_name = None

        # Curriculum tracking
        self.curriculum_stage  = 0
        self.episode_results   = []   # rolling window of True/False (parked?)
        self.window_size       = 20

        self._build_world()
        self._build_ui()
        self.picker = ModelPicker(self.screen, self.font_s, self.font_m)
        self.reset()

    # ── World ─────────────────────────────────────────────────────────────────
    def _build_world(self):
        segs = []
        # Screen boundary walls
        segs += [((0,0),(SCREEN_W,0)),((SCREEN_W,0),(SCREEN_W,SCREEN_H)),
                 ((SCREEN_W,SCREEN_H),(0,SCREEN_H)),((0,SCREEN_H),(0,0))]

        if self.curriculum_stage < 4:
            # ── Stages 1-4: vertical bay layout ──────────────────────────────
            for bx,by in BAYS:
                segs += [((bx,by),(bx,by+BAY_H)),((bx+BAY_W,by),(bx+BAY_W,by+BAY_H))]
            segs += [((0,BAY_ROW_Y+BAY_H),(SCREEN_W,BAY_ROW_Y+BAY_H))]
            for idx in OBSTACLE_IDXS:
                bx,by = BAYS[idx]
                cx,cy = bx+BAY_W/2, by+BAY_H/2
                segs += segments_from_rect(cx, cy, CAR_W+2, CAR_H+2)
            tbx,tby = BAYS[TARGET_IDX]
            self.target_rect  = pygame.Rect(tbx, tby, BAY_W, BAY_H)
            self.target_cx    = tbx + BAY_W/2
            self.target_cy    = tby + BAY_H/2
            self.target_angle = math.pi

        else:
            # ── Stage 5: horizontal parallel parking layout ───────────────────
            # Parallel bays run along the yellow lane line (BAY_ROW_Y)
            # Bay is rotated: width=BAY_H(90), height=BAY_W(50) — long side horizontal
            PB_W  = 90   # parallel bay width  (along road)
            PB_H  = 50   # parallel bay height (into kerb)
            PB_Y  = BAY_ROW_Y   # top of parallel bays (just below lane line)
            PB_CX = SCREEN_W // 2   # centre bay x

            # Target bay
            self.target_cx    = float(PB_CX)
            self.target_cy    = float(PB_Y + PB_H / 2)
            self.target_rect  = pygame.Rect(PB_CX - PB_W//2, PB_Y, PB_W, PB_H)
            self.target_angle = math.pi / 2   # car must face right (horizontal)

            # Obstacle cars in bays on either side
            for off in (-PB_W - 6, PB_W + 6):
                ocx = PB_CX + off
                ocy = PB_Y + PB_H / 2
                segs += segments_from_rect(ocx, ocy, CAR_H+2, CAR_W+2)  # rotated

            # Kerb wall below bays
            segs += [((0, PB_Y + PB_H),(SCREEN_W, PB_Y + PB_H))]

            # Store for render
            self._pb_w  = PB_W
            self._pb_h  = PB_H
            self._pb_y  = PB_Y
            self._pb_cx = PB_CX

        self.world_segs = segs

    def _build_ui(self):
        bx = SCREEN_W - 215
        self.btn_mode = Button(bx,  14, 200, 30, "Mode: HUMAN", COL_BTN, COL_BTN_ACTIVE)
        self.btn_load = Button(bx,  52, 200, 30, "Load Model",  COL_BTN)
        self.buttons  = [self.btn_mode, self.btn_load]

    # ── Spawn helpers ─────────────────────────────────────────────────────────
    def _spawn_position(self):
        stage = CURRICULUM[self.curriculum_stage]
        cx    = self.target_cx
        cy    = self.target_cy

        if self.curriculum_stage == 4:
            # Stage 5 — parallel parking
            # Spawn car horizontal, near the lane line, offset to the right of the bay
            sx = float(self.target_cx + np.random.uniform(120, 250))
            sx = float(np.clip(sx, 100, SCREEN_W - 100))
            sy = float(self.target_cy + np.random.uniform(-15, 15))
            sa = math.pi / 2   # facing right (horizontal), same as target
            return sx, sy, sa

        if stage["fixed_offset"] is not None:
            # Fixed spawn: exact centre above the bay, no randomness at all
            sx = cx
            sy = cy - stage["fixed_offset"]   # directly above target
        else:
            sx = cx + np.random.uniform(-stage["x_rad"], stage["x_rad"])
            sx = float(np.clip(sx, 60, SCREEN_W-60))
            sy = float(np.random.uniform(stage["y_min"], stage["y_max"]))

        # Heading
        if self.curriculum_stage in (2, 3):
            # Reverse stages: car faces upward, will reverse down into bay
            sa = 0.0
        else:
            # Forward stages: car faces straight down toward bay
            sa = math.pi

        return float(sx), float(sy), sa

    # ── Gym API ───────────────────────────────────────────────────────────────
    def reset(self):
        self._build_world()   # rebuild so layout matches current stage
        sx, sy, sa   = self._spawn_position()
        self.agent   = Car(sx, sy, sa)
        self.steps   = 0
        self.done    = False
        self.status  = ""
        self.collision = False
        self.reward_last = 0.0
        self.prev_dist = math.hypot(self.target_cx-sx, self.target_cy-sy)
        return self._obs()

    def step(self, action):
        throttle, brake, steer, gear = action
        self.agent.update(throttle, brake, steer, gear)
        self.steps += 1
        obs          = self._obs()
        reward, done = self._reward_done()
        self.reward_last = reward
        self.done        = done
        return obs, reward, done

    def _obs(self):
        a  = self.agent
        dx = self.target_cx - a.x
        dy = self.target_cy - a.y
        dist = math.hypot(dx, dy)
        angle_to_target = math.atan2(dx, -dy) - a.angle
        angle_diff = ((a.angle - self.target_angle)+math.pi)%(2*math.pi) - math.pi
        rev_diff   = ((a.angle - self.target_angle + math.pi)+math.pi)%(2*math.pi) - math.pi
        best_diff  = angle_diff if abs(angle_diff) < abs(rev_diff) else rev_diff
        rays = a.sense(self.world_segs)
        return np.array([
            a.x/SCREEN_W, a.y/SCREEN_H,
            math.sin(a.angle), math.cos(a.angle),
            a.speed/MAX_SPEED, a.steer/MAX_STEER,
            dist/SCREEN_W,
            math.sin(angle_to_target), math.cos(angle_to_target),
            best_diff/math.pi,
            float(a.gear)
        ] + [r/RAY_LEN for r in rays], dtype=np.float32)

    def _reward_done(self):
        a = self.agent
        ax1,ay1,ax2,ay2 = poly_aabb(a.corners())

        # Collision
        self.collision = False
        for idx in OBSTACLE_IDXS:
            bx,by = BAYS[idx]
            pcx,pcy = bx+BAY_W/2, by+BAY_H/2
            pr = pygame.Rect(pcx-CAR_W/2-1, pcy-CAR_H/2-1, CAR_W+2, CAR_H+2)
            if pr.colliderect(pygame.Rect(ax1,ay1,ax2-ax1,ay2-ay1)):
                self.collision = True

        oob = (a.x<5 or a.x>SCREEN_W-5 or a.y<5 or a.y>BAY_ROW_Y+BAY_H+5)

        dx   = self.target_cx - a.x
        dy   = self.target_cy - a.y
        dist = math.hypot(dx, dy)

        # Angle — accept both forward and reverse parking
        fwd_diff = abs(((a.angle - self.target_angle)+math.pi)%(2*math.pi) - math.pi)
        rev_diff = abs(((a.angle - self.target_angle + math.pi)+math.pi)%(2*math.pi) - math.pi)
        best_angle = min(fwd_diff, rev_diff)

        in_bay  = self.target_rect.collidepoint(a.x, a.y)
        stopped = abs(a.speed) < 0.05
        parked  = in_bay and dist < 20 and best_angle < 0.35 and stopped

        # ── Reward shaping ────────────────────────────────────────────────────
        # Progress reward — reward getting closer each step
        delta_dist  = self.prev_dist - dist
        self.prev_dist = dist

        reward  = delta_dist * 0.5           # strong pull toward target

        # Gear shaping — strongly enforce correct gear per stage
        fwd_stage = self.curriculum_stage in (0, 1)   # Stage 1 & 2
        rev_stage = self.curriculum_stage in (2, 3)   # Stage 3 & 4
        if fwd_stage and a.gear == -1:
            reward -= 0.08   # discourage reversing in forward stages
        if rev_stage and not in_bay:
            if a.gear == 1:
                reward -= 0.25   # heavy penalty for going forward in reverse stages
            else:
                reward += 0.05   # small bonus each step for correctly using reverse
            # Bonus for nose pointing AWAY from bay (correct reverse orientation)
            # In reverse stages target_angle=pi (down), so car facing UP (angle~0) is correct
            nose_away = abs((a.angle % (2*math.pi)))  # how close to 0 (pointing up)
            if nose_away > math.pi: nose_away = 2*math.pi - nose_away
            if nose_away < math.pi/3:   # within 60 degrees of pointing up
                reward += 0.08
        reward -= best_angle * 0.15          # facing right direction
        reward -= 0.005                      # small time penalty
        reward -= 10.0 if self.collision else 0
        reward -= 10.0 if oob            else 0
        reward += 30.0 if parked         else 0
        # Bonus for being close and aligned but not yet stopped
        if in_bay and best_angle < 0.5:
            reward += 1.0

        # Done
        if parked:
            self.status = "PARKED!"; done = True
            self.episode_results.append(True)
        elif self.collision:
            self.status = "CRASHED!"; done = True
            self.episode_results.append(False)
        elif oob:
            self.status = "OUT OF BOUNDS"; done = True
            self.episode_results.append(False)
        elif self.steps > 1200:
            self.status = "TIME OUT"; done = True
            self.episode_results.append(False)
        else:
            done = False

        # Keep rolling window
        if len(self.episode_results) > self.window_size:
            self.episode_results.pop(0)

        # Curriculum advancement
        if done and len(self.episode_results) == self.window_size:
            rate = sum(self.episode_results) / self.window_size
            stage = CURRICULUM[self.curriculum_stage]
            if rate >= stage["target"] and self.curriculum_stage < len(CURRICULUM)-1:
                self.curriculum_stage += 1
                self.episode_results  = []
                print(f"[CURRICULUM] Advanced to {CURRICULUM[self.curriculum_stage]['name']}")

        return reward, done

    # ── Render ────────────────────────────────────────────────────────────────
    def render(self):
        s = self.screen
        s.fill(COL_ASPHALT)

        pygame.draw.line(s, COL_LANE, (0, BAY_ROW_Y-8), (SCREEN_W, BAY_ROW_Y-8), 2)

        if self.curriculum_stage < 4:
            # ── Stages 1-4: vertical bays ─────────────────────────────────────
            for i,(bx,by) in enumerate(BAYS):
                if i == TARGET_IDX:
                    surf = pygame.Surface((BAY_W, BAY_H), pygame.SRCALPHA)
                    surf.fill((*COL_BAY_TARGET, 45))
                    s.blit(surf, (bx, by))
                    pygame.draw.rect(s, COL_BAY_TARGET, (bx,by,BAY_W,BAY_H), 2)
                    lbl = self.font_s.render("P", True, COL_BAY_TARGET)
                    s.blit(lbl, (bx+BAY_W//2-5, by+BAY_H//2-7))
                elif i in OBSTACLE_IDXS:
                    cx,cy = bx+BAY_W//2, by+BAY_H//2
                    pygame.draw.polygon(s, COL_CAR_PARKED, rect_corners(cx,cy,CAR_W,CAR_H,0))
                    pygame.draw.polygon(s, (60,65,75), rect_corners(cx,cy,CAR_W,CAR_H,0), 1)
                else:
                    pygame.draw.rect(s, COL_BAY_IDLE, (bx,by,BAY_W,BAY_H), 1)
        else:
            # ── Stage 5: horizontal parallel bays ─────────────────────────────
            PB_W  = self._pb_w
            PB_H  = self._pb_h
            PB_Y  = self._pb_y
            PB_CX = self._pb_cx

            # Target bay (horizontal green rectangle)
            surf = pygame.Surface((PB_W, PB_H), pygame.SRCALPHA)
            surf.fill((*COL_BAY_TARGET, 45))
            s.blit(surf, (PB_CX - PB_W//2, PB_Y))
            pygame.draw.rect(s, COL_BAY_TARGET, (PB_CX-PB_W//2, PB_Y, PB_W, PB_H), 2)
            lbl = self.font_s.render("P", True, COL_BAY_TARGET)
            s.blit(lbl, (PB_CX-5, PB_Y+PB_H//2-7))

            # Obstacle cars on either side (drawn horizontal)
            for off in (-PB_W - 6, PB_W + 6):
                ocx = PB_CX + off
                ocy = PB_Y + PB_H // 2
                # Rotated 90deg: swap CAR_W/CAR_H so car lies flat
                pygame.draw.polygon(s, COL_CAR_PARKED,
                                    rect_corners(ocx, ocy, CAR_H, CAR_W, 0))
                pygame.draw.polygon(s, (60,65,75),
                                    rect_corners(ocx, ocy, CAR_H, CAR_W, 0), 1)

            # Empty bay outlines either side of obstacles
            for off in (-3*(PB_W+6), 3*(PB_W+6)):
                pygame.draw.rect(s, COL_BAY_IDLE,
                                 (PB_CX+off-PB_W//2, PB_Y, PB_W, PB_H), 1)

        # Rays
        a = self.agent
        rays = a.sense(self.world_segs)
        for i, dist in enumerate(rays):
            ang = a.angle + math.radians(i*(360/NUM_RAYS))
            ex  = a.x + math.cos(ang)*dist
            ey  = a.y + math.sin(ang)*dist
            pygame.draw.line(s, (*COL_SENSOR, 80), (int(a.x),int(a.y)), (int(ex),int(ey)), 1)
            col_dot = COL_HIT if dist < RAY_LEN*0.25 else COL_SENSOR
            pygame.draw.circle(s, col_dot, (int(ex),int(ey)), 3)

        # Car
        col_car = COL_HIT if self.collision else COL_CAR_AGENT
        corners = a.corners()
        pygame.draw.polygon(s, col_car, corners)
        pygame.draw.polygon(s, (150,180,255), corners, 1)
        fm = ((corners[0][0]+corners[1][0])/2, (corners[0][1]+corners[1][1])/2)
        pygame.draw.line(s, (150,200,255), (int(a.x),int(a.y)), (int(fm[0]),int(fm[1])), 2)

        # ── HUD ───────────────────────────────────────────────────────────────
        stage_name = CURRICULUM[self.curriculum_stage]["name"]
        if self.episode_results:
            rate = sum(self.episode_results)/len(self.episode_results)
            target = CURRICULUM[self.curriculum_stage]["target"]
            rate_str = f"{rate:.0%} / {target:.0%}"
        else:
            rate_str = "—"

        hud = [
            f"step     {self.steps:>5}",
            f"speed    {a.speed:>+6.2f}",
            f"steer    {math.degrees(a.steer):>+6.1f}°",
            f"gear     {'FWD' if a.gear==1 else 'REV'}",
            f"reward   {self.reward_last:>+6.2f}",
            f"stage    {self.curriculum_stage+1}/5",
            f"success  {rate_str}",
        ]
        py = 14
        for line in hud:
            s.blit(self.font_s.render(line, True, COL_TEXT), (14, py))
            py += 18

        # Stage name
        sn = self.font_s.render(stage_name, True, COL_BAY_TARGET)
        s.blit(sn, (14, py+4))

        if self.loaded_name:
            ln = self.font_s.render(f"model: {self.loaded_name}", True, (100,180,130))
            s.blit(ln, (14, py+22))

        # Controls
        if self.agent_mode == "human":
            ctrl = ["W/S  throttle/brake","A/D  steer","R    reverse","SPACE reset"]
            px2,py2 = SCREEN_W-215, 100
            for line in ctrl:
                s.blit(self.font_s.render(line, True, (100,110,130)), (px2,py2))
                py2 += 18

        # Buttons
        mp = pygame.mouse.get_pos()
        for btn in self.buttons: btn.check_hover(mp)
        self.btn_mode.label  = f"Mode: {'RL AGENT' if self.agent_mode=='rl' else 'HUMAN'}"
        self.btn_mode.active = self.agent_mode == "rl"
        for btn in self.buttons: btn.draw(s, self.font_s)

        # Status banner
        if self.done and self.status:
            col = (COL_BAY_TARGET if self.status=="PARKED!" else
                   COL_LANE       if self.status=="TIME OUT" else COL_HIT)
            txt = self.font_m.render(f"── {self.status} ── press SPACE", True, col)
            s.blit(txt, (SCREEN_W//2-txt.get_width()//2, SCREEN_H//2-12))

        # File picker on top
        self.picker.draw()
        pygame.display.flip()

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        gear = 1
        obs  = self.reset()

        while True:
            self.clock.tick(FPS)
            mp = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return

                # Let picker consume events first
                if self.picker.visible:
                    result = self.picker.handle_event(event)
                    if result and result != "cancel":
                        self._load_model(result)
                    continue

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        obs = self.reset(); gear = 1
                    if event.key == pygame.K_r and self.agent_mode=="human":
                        gear *= -1

                if self.btn_mode.is_clicked(mp, event):
                    self.agent_mode = "rl" if self.agent_mode=="human" else "human"
                    obs = self.reset(); gear = 1

                if self.btn_load.is_clicked(mp, event):
                    self.picker.open()

            if not self.done and not self.picker.visible:
                if self.agent_mode == "human":
                    keys = pygame.key.get_pressed()
                    throttle = 1.0 if keys[pygame.K_w] else 0.0
                    brake    = 1.0 if keys[pygame.K_s] else 0.0
                    steer    = (-1.0 if keys[pygame.K_a] else
                                 1.0 if keys[pygame.K_d] else 0.0)
                    obs, _, _ = self.step((throttle, brake, steer, gear))

                elif self.agent_mode=="rl" and self.rl_agent is not None:
                    action, _ = self.rl_agent.predict(obs, deterministic=True)
                    throttle  = float(np.clip( action[0], 0, 1))
                    brake     = float(np.clip(-action[0], 0, 1))
                    steer     = float(np.clip( action[1], -1, 1))
                    gear      = -1 if (len(action)>2 and action[2]<0) else 1
                    obs, _, done = self.step((throttle, brake, steer, gear))
                    if done: obs = self.reset()

            self.render()

    def _load_model(self, path):
        try:
            from stable_baselines3 import PPO
            self.rl_agent    = PPO.load(path)
            self.loaded_name = os.path.basename(path)
            print(f"[ENV] Loaded: {self.loaded_name}")
        except Exception as e:
            self.loaded_name = f"Error loading model"
            print(f"[ENV] Error: {e}")


if __name__ == "__main__":
    env = ParkingEnv()
    env.run()