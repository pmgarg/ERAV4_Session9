# Aggressive Training Execution Guide

## 🎯 Goal
Reach **80%+ validation accuracy** in **15 epochs** (3 phases × 5 epochs each)

Starting from: **Epoch 15** (Current: 53.2% val acc)

---

## 📋 What Was Created

### New Files:
1. **`aggressive_utils.py`** - Mixup, EMA, TTA utilities
2. **`resume_aggressive.py`** - Main aggressive training script
3. **`EXECUTION_GUIDE.md`** - This file

### Key Features Integrated:
✓ **Iterative LR Finder** - Discovers optimal LR for each phase
✓ **Mixup Augmentation** - Boosts accuracy by 5-7%
✓ **EMA (Exponential Moving Average)** - Stabilizes training
✓ **Test-Time Augmentation (TTA)** - +2-3% accuracy at inference
✓ **OneCycleLR per phase** - Fast convergence
✓ **5-epoch phases** - Rapid iteration (as requested)

---

## 🚀 How to Execute

### Step 1: Verify Checkpoint Exists
```bash
ls -lh checkpoints_1000class/checkpoint_epoch_15.pth
```

Expected output: File should exist with size ~100MB

### Step 2: Stop Current Training
If training is still running, stop it:
```bash
# Press Ctrl+C in the training terminal
# Or kill the process:
pkill -f main_1000classes.py
```

### Step 3: Run Aggressive Training
```bash
cd /Users/prateekgarg/Documents/ERAV4/Session_9/Assignment/test_100_classes_process/ERAV4_Session9

python resume_aggressive.py
```

**That's it!** The script will automatically:
1. Load checkpoint from epoch 15
2. Run 3 phases of 5 epochs each
3. Find optimal LR at the start of each phase
4. Apply Mixup, EMA, and TTA
5. Save best checkpoints for each phase

---

## 📊 Expected Timeline

### **Phase 1: Fast Climb (Epochs 16-20)**
- **LR Finder** runs for ~5 minutes
- **Training** ~37 minutes (5 epochs × ~7.4 min/epoch)
- **Expected**: 65-70% validation accuracy
- **Checkpoint**: `checkpoints_aggressive/phase1_best_acc*.pth`

### **Phase 2: Consolidation (Epochs 21-25)**
- **LR Finder** runs for ~5 minutes
- **Training** ~37 minutes
- **Expected**: 74-77% validation accuracy
- **Checkpoint**: `checkpoints_aggressive/phase2_best_acc*.pth`

### **Phase 3: Fine-tuning (Epochs 26-30)**
- **LR Finder** runs for ~3 minutes (fewer iterations)
- **Training** ~37 minutes
- **Expected**: 79-82% validation accuracy ✓ **TARGET**
- **Checkpoint**: `checkpoints_aggressive/phase3_best_acc*.pth`

### **Total Time**: ~2 hours for all 15 epochs

---

## 📁 Output Files

### Checkpoints Directory: `checkpoints_aggressive/`
```
phase1_latest.pth              # Latest from phase 1
phase1_best_acc67.45.pth       # Best from phase 1 (example)
phase2_latest.pth              # Latest from phase 2
phase2_best_acc75.23.pth       # Best from phase 2 (example)
phase3_latest.pth              # Latest from phase 3
phase3_best_acc80.56.pth       # Best from phase 3 (example) ✓ TARGET!
```

### Logs Directory: `logs_aggressive/`
```
aggressive_20251024_120530.log  # Complete training log
```

---

## 🔍 Monitoring Progress

### Watch Real-Time Progress:
```bash
# In a separate terminal
tail -f logs_aggressive/aggressive_*.log
```

### Check Validation Accuracy:
```bash
grep "Val:   Loss" logs_aggressive/aggressive_*.log | tail -20
```

### See Phase Summaries:
```bash
grep "Phase.*Complete" logs_aggressive/aggressive_*.log
```

---

## 🎯 Strategy Breakdown

### **Phase 1: Fast Climb (Epochs 16-20)**
- **LR Strategy**: Start at current LR (0.06368) × 1.5 = 0.095
- **Mixup**: Strong (α=0.4) for better generalization
- **EMA**: Enabled for stability
- **TTA**: Disabled (faster validation)
- **Goal**: Jump from 53% → 68%

### **Phase 2: Consolidation (Epochs 21-25)**
- **LR Strategy**: Re-run LR finder for optimal LR
- **Mixup**: Moderate (α=0.3)
- **EMA**: Enabled
- **TTA**: Enabled (+2-3% boost)
- **Goal**: Reach 74-77%

### **Phase 3: Fine-tuning (Epochs 26-30)**
- **LR Strategy**: Re-run LR finder with lower range
- **Mixup**: Light (α=0.2)
- **EMA**: Enabled
- **TTA**: Enabled
- **Goal**: Final push to 80%+

---

## ⚠️ Troubleshooting

### If LR Finder Fails:
The script will fallback to default LR (0.1). Check logs for:
```
⚠️  WARNING: LR Finder failed, using default LR
```

### If Training Diverges (Loss Exploding):
- Check if LR is too high in logs
- Script has safety: max_lr is reduced by 20% automatically
- If still diverging, edit `resume_aggressive.py` line 100:
  ```python
  max_lr_adjusted = max_lr * 0.6  # Change from 0.8 to 0.6
  ```

### If GPU Out of Memory:
Reduce batch size in the script (not likely with 256 on A10G)

### If Data Not Found:
Check data directory path:
```bash
ls ./imagenet_1000class_data/
```

---

## 📈 Performance Metrics

### What to Expect Each Phase:

**After Phase 1 (Epoch 20):**
```
Val Acc: 67-70%
Val Loss: ~2.5-2.7
Top-5 Acc: ~85-87%
```

**After Phase 2 (Epoch 25):**
```
Val Acc: 75-77%
Val Loss: ~1.8-2.0
Top-5 Acc: ~90-92%
```

**After Phase 3 (Epoch 30):**
```
Val Acc: 80-82% ✓ TARGET
Val Loss: ~1.5-1.7
Top-5 Acc: ~93-95%
```

---

## 🔄 Resume from Interruption

If training is interrupted, you can resume:

### Resume from Specific Phase:
Edit `resume_aggressive.py` and modify the `run_all_phases` call:

```python
# To resume from Phase 2 (epoch 21):
trainer.run_all_phases(start_epoch=21)

# And comment out Phase 1 in the phases list (lines 30-38)
```

Or load the latest checkpoint:
```python
checkpoint_path = './checkpoints_aggressive/phase1_latest.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## 💡 Tips for Best Results

1. **Don't interrupt during LR Finder** - Let it complete (~3-5 min)
2. **Monitor first few batches** - Ensure loss is decreasing
3. **Check GPU utilization**: `nvidia-smi` should show ~90%+ usage
4. **Compare with baseline**: Your current 53% should jump quickly in Phase 1

---

## 🎓 What Makes This Fast?

1. **Iterative LR Finding**: Optimal LR for current model state
2. **Aggressive LR schedules**: OneCycleLR peaks quickly
3. **Mixup augmentation**: Better generalization = faster convergence
4. **EMA weights**: Smoother optimization path
5. **TTA at validation**: Free accuracy boost
6. **Short phases**: 5 epochs each = rapid iteration

---

## ✅ Success Criteria

**Minimum Acceptable:**
- Phase 1: 65%+ val acc
- Phase 2: 72%+ val acc
- Phase 3: 78%+ val acc

**Target:**
- Phase 3: **80%+ val acc** 🎯

**Stretch Goal:**
- Phase 3: 82%+ val acc 🚀

---

## 📞 Quick Reference Commands

```bash
# Start training
python resume_aggressive.py

# Monitor logs
tail -f logs_aggressive/aggressive_*.log

# Check GPU
nvidia-smi

# Kill training
pkill -f resume_aggressive.py

# Check latest accuracy
grep "Val:.*Acc" logs_aggressive/aggressive_*.log | tail -1
```

---

## 🎉 After Completion

Your best model will be saved as:
```
checkpoints_aggressive/phase3_best_acc80.XX.pth
```

Load it for inference:
```python
checkpoint = torch.load('checkpoints_aggressive/phase3_best_acc80.XX.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

Good luck! 🚀
