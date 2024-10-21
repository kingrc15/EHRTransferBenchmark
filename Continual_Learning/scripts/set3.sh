# Decompensation: (1 epoch, 100k:100k:100k:50k:25k samples)   
# --------------------------------#
python3 ../tests/test.py --region 1 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5           # EWC
python3 ../tests/test.py --region 2 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5           # EWC
python3 ../tests/test.py --region 3 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5           # EWC
python3 ../tests/test.py --region 4 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5           # EWC

python3 ../tests/test.py --region 1 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined
python3 ../tests/test.py --region 2 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined
python3 ../tests/test.py --region 3 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined
python3 ../tests/test.py --region 4 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined

# LoS: (1 epoch, 100k:100k:100k:50k:25k samples)
# --------------------------------#
python3 ../tests/test.py --region 1 --tasks 2 --los --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined
python3 ../tests/test.py --region 2 --tasks 2 --los --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined
python3 ../tests/test.py --region 3 --tasks 2 --los --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined
python3 ../tests/test.py --region 4 --tasks 2 --los --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined