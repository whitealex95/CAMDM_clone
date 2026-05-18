"""
Filter the env2d-augmented pkl so that jump/door styles keep only the
'none' obstacle mode (walk retains every mode produced by create_env_dataset).

Why: jump/door motions are short clips not intended for obstacle-avoidance
conditioning, so we don't want them duplicated with detour-style obstacles.

Input:  data/pkls/lafan1_g1_motion30_jump_door_env2d_detour_only_none.pkl
Output: overwrites the same path (the canonical training dataset).
"""
import argparse
import pickle


DEFAULT_PATH = "data/pkls/lafan1_g1_motion30_jump_door_env2d_detour_only_none.pkl"
STYLES_NONE_ONLY = {"jump", "door"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=DEFAULT_PATH,
                   help="Augmented env2d pkl to filter in place.")
    args = p.parse_args()

    with open(args.path, "rb") as f:
        data = pickle.load(f)

    kept, dropped = [], []
    for m in data["motions"]:
        if m["style"] in STYLES_NONE_ONLY and m["obstacle_mode"] != "none":
            dropped.append(m)
        else:
            kept.append(m)

    data["motions"] = kept

    with open(args.path, "wb") as f:
        pickle.dump(data, f)

    print(f"Filtered {args.path}")
    print(f"  kept    : {len(kept)} clips")
    print(f"  dropped : {len(dropped)} clips")
    for m in kept:
        T = m["local_joint_rotations"].shape[0]
        print(f"    style={m['style']:<5} mode={m['obstacle_mode']:<12} T={T:5d}")


if __name__ == "__main__":
    main()
