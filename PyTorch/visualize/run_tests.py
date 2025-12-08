#!/usr/bin/env python3
"""
Run all visualization steps in sequence for testing.
This helps verify each component before moving to the next.
"""

import os
import sys
import subprocess

def run_step(step_num, script_name, description, args=None):
    """Run a visualization step and wait for user confirmation."""
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {description}")
    print("=" * 70)
    
    cmd = ["python", script_name]
    if args:
        cmd.extend(args)
    
    print(f"\nCommand: {' '.join(cmd)}")
    print("\nPress ENTER to run this step (or 's' to skip)...")
    
    response = input().strip().lower()
    if response == 's':
        print("Skipped.")
        return None
    
    print("\nRunning...\n")
    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return -1
    except Exception as e:
        print(f"\nError running step: {e}")
        return 1


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   CAMDM Motion Visualization - Step-by-Step Testing                 ║
║                                                                      ║
║   This script will run each visualization step in sequence.         ║
║   You can verify each step works before moving to the next.         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    steps = [
        {
            "num": 1,
            "script": "visualize/step1_test_mujoco.py",
            "description": "Test MuJoCo Model Loading",
            "args": None,
            "verify": """
What to check:
- MuJoCo viewer opens
- G1 robot appears in standing pose
- Joint order verification passes
- qpos size is 36 DOF
- Press ESC to close viewer when done
            """
        },
        {
            "num": 2,
            "script": "visualize/test_loader.py",
            "description": "Test Motion Data Loading",
            "args": None,
            "verify": """
What to check:
- Finds .pkl files in data/pkls/
- Loads dataset successfully
- Shows motion summary
- qpos dimensions are correct (36,)
- No errors or exceptions
            """
        },
        {
            "num": 3,
            "script": "visualize/step2_visualize_data.py",
            "description": "Visualize Training Data (lafan1_g1)",
            "args": ["--dataset", "lafan1_g1"],
            "verify": """
What to check:
- Viewer opens with animated robot
- Motion plays smoothly at 30 FPS
- Robot movements look natural
- Test controls:
  * SPACE: pause/resume
  * UP/DOWN: switch motions
  * LEFT/RIGHT: step frames
  * R: reset
  * 1-9: speed control
- No joint violations or glitches
- Press ESC to close when done
            """
        },
    ]
    
    results = []
    
    for step in steps:
        print(step["verify"])
        
        returncode = run_step(
            step["num"],
            step["script"],
            step["description"],
            step["args"]
        )
        
        if returncode is None:
            results.append(("SKIPPED", step["num"], step["description"]))
            continue
        
        if returncode != 0:
            print(f"\n⚠ Step {step['num']} exited with code {returncode}")
            print("\nDo you want to:")
            print("  c - Continue to next step")
            print("  r - Re-run this step")
            print("  q - Quit")
            
            while True:
                choice = input("\nChoice: ").strip().lower()
                if choice == 'c':
                    results.append(("FAILED", step["num"], step["description"]))
                    break
                elif choice == 'r':
                    returncode = run_step(
                        step["num"],
                        step["script"],
                        step["description"],
                        step["args"]
                    )
                    if returncode == 0:
                        results.append(("PASSED", step["num"], step["description"]))
                        break
                elif choice == 'q':
                    print("\nExiting...")
                    return
        else:
            results.append(("PASSED", step["num"], step["description"]))
        
        print(f"\n✓ Step {step['num']} completed")
        input("Press ENTER to continue to next step...")
    
    # Summary
    print("\n" + "=" * 70)
    print("TESTING SUMMARY")
    print("=" * 70)
    
    for status, num, desc in results:
        symbol = "✓" if status == "PASSED" else "✗" if status == "FAILED" else "⊘"
        print(f"{symbol} Step {num}: {desc} - {status}")
    
    print("\n" + "=" * 70)
    
    passed = sum(1 for s, _, _ in results if s == "PASSED")
    total = len(results)
    
    if passed == total:
        print(f"All {total} steps passed! 🎉")
        print("\nYou're ready to move on to:")
        print("  - Step 3: Visualize generated motions")
        print("  - Step 4: Interactive control")
    else:
        print(f"{passed}/{total} steps passed.")
        print("\nPlease review and fix failed steps before proceeding.")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
        sys.exit(1)
