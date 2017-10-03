# Complete the function below.
def run_base(bases=[]):
    base_copy = bases[:]
    if bases[0] == 0:
        bases[0] = 1
    else:
        for i in range(1,len(bases)):
            bases[i] = base_copy[i-1]
    print bases

def totalRuns(batterSpeedLimits, pitchSpeeds):
    run_count = 0
    bases = [0,0,0,0]
    n = len(batterSpeedLimits)
    for i in range(0,n):
        for pitch in pitchSpeeds[i]:
            if pitch <= batterSpeedLimits[i]:
                run_base(bases)
                if bases[3] == 1:
                    run_count += 1
                break

    return run_count


print totalRuns([70,80,90,100],[[72,71,70],[82,81,80],[92,91,90],[100,101,102]])


