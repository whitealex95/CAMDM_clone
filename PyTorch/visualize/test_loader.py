"""
Quick test to verify motion_loader.py works with your pickle files
Run this before Step 2 to ensure data loads correctly.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualize.motion_loader import MotionDataset

def test_motion_loader():
    print("=" * 60)
    print("Testing Motion Loader")
    print("=" * 60)
    
    # Check available datasets
    pkl_dir = "data/pkls"
    if not os.path.exists(pkl_dir):
        print(f"✗ Directory not found: {pkl_dir}")
        return False
    
    pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]
    if not pkl_files:
        print(f"✗ No .pkl files found in {pkl_dir}")
        return False
    
    print(f"\nFound {len(pkl_files)} dataset(s):")
    for pkl_file in pkl_files:
        print(f"  - {pkl_file}")
    
    # Test loading first dataset
    test_pkl = os.path.join(pkl_dir, pkl_files[0])
    print(f"\nTesting with: {test_pkl}")
    print("-" * 60)
    
    try:
        dataset = MotionDataset(test_pkl)
        dataset.print_summary()
        
        # Test accessing first motion
        print("Testing motion data access...")
        motion = dataset[0]
        print(f"✓ Loaded motion: {motion}")
        
        # Test qpos extraction
        qpos = motion.get_qpos(0)
        print(f"✓ qpos shape: {qpos.shape}")
        print(f"  - Root position: {qpos[:3]}")
        print(f"  - Root quaternion (WXYZ): {qpos[3:7]}")
        print(f"  - First 5 joint angles: {qpos[7:12]}")
        
        # Verify dimensions
        assert qpos.shape == (36,), f"Expected shape (36,), got {qpos.shape}"
        print(f"✓ qpos dimensions correct (36 DOF)")
        
        # Test all qpos
        all_qpos = motion.get_all_qpos()
        print(f"✓ All frames qpos shape: {all_qpos.shape}")
        assert all_qpos.shape == (motion.num_frames, 36)
        
        print("\n" + "=" * 60)
        print("✓ Motion loader test PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Motion loader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_motion_loader()
    sys.exit(0 if success else 1)
