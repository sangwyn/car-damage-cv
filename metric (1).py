import torch

parts = ["Quarter-panel", "Front-wheel", "Back-window", "Trunk", "Front-door", "Rocker-panel",
         "Grille", "Windshield", "Front-window", "Back-door", "Headlight", "Back-wheel", "Back-windshield",
         "Hood", "Fender", "Tail-light", "License-plate", "Front-bumper", "Back-bumper", "Mirror", "Roof"]

dmgs = ["Missing part", "Broken part", "Scratch", "Cracked", "Dent", "Flaking", "Paint chip", "Corrosion"]

part_weights = {
    "Windshield": 10,       # Safety-critical
    "Back-windshield": 9,   # Structural/safety importance
    "Headlight": 8,         # Visibility/safety
    "Tail-light": 8,
    "Front-wheel": 8,       # Critical for movement
    "Back-wheel": 8,
    "Mirror": 8,
    "Front-window": 8,      # Side windows
    "Hood": 7,              # Engine protection
    "Roof": 7,              # Structural
    "Back-window": 7,
    "Front-bumper": 6,      # Impact absorption
    "Back-bumper": 6,
    "Front-door": 6,        # Safety/function
    "Back-door": 6,
    "Trunk": 6,             # Storage/water ingress risk
    "Quarter-panel": 5,     # Cosmetic/repair cost
    "Fender": 5,
    "Rocker-panel": 5,
    "Grille": 4,            # Cosmetic/minor function
    "License-plate": 2      # Cosmetic/easy replacement
}

dmg_weights = {
    "Missing part": 10,   # Complete failure
    "Broken part": 9,     # Severe damage
    "Cracked": 8,         # Structural compromise
    "Dent": 7,            # Affects structure/aerodynamics
    "Flaking": 6,         # Leads to corrosion
    "Corrosion": 5,       # Long-term degradation
    "Paint chip": 4,      # Cosmetic/minor
    "Scratch": 3          # Superficial
}

# Изображения для проверки: 100 (4), 101 (4), 103 (2), 2 (1), 1053 (5), 1045 (3), 1046 (1), 1057 (4) -- only windshield

def get_grade(part_mask: torch.Tensor, damage_mask: torch.Tensor):
    global parts, dmgs, part_weights, dmg_weights
    
    total_score = 0

    x_min, y_min, x_max, y_max = 0, 0, part_mask.shape[2], part_mask.shape[1]
    part_pixels = {}
    for i in range(part_mask.shape[1]):
        for j in range(part_mask.shape[2]):
            for k, part in enumerate(parts):
                if part_mask[k, i, j] == 0:
                    continue

                x_min, y_min = min(x_min, j), min(y_min, i)
                x_max, y_max = max(x_max, j), max(y_max, i)

                if part not in part_pixels:
                    part_pixels[part] = {'total': 0}
                    for dmg in dmg_types:
                        part_pixels[part][dmg] = 0

                for l, dmg in enumerate(dmgs):
                    if damage_mask[l, i, j] == 0:
                        continue
                    
                    part_pixels[part][dmg] += 1
                    part_pixels[part]['total'] += 1

    for part in parts:
        for dmg in dmgs:
            dmg_area = part_pixels[part][dmg] / part_pixels[part]['total']
            total_score += part_weights[part] * dmg_weights[dmg] * dmg_area

    car_area = (x_max - x_min + 1) * (y_max - y_min + 1)
    max_possible_weight = max(max(part_weights.values()), max(dmg_weights.values()))

    normalized_score = (total_score / (car_area * max_possible_weight)) * 100

    grade = normalized_score // 20 + 1

    return normalized_score, grade