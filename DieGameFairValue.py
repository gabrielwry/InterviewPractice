def die_game_fair_value(rolls):
    if rolls > 100:
        return 59999
    return helper(1,rolls,3.5)

def helper(n,N,fair):
    if n == N:
        return fair
    odds_to_stop = int(7-fair)/6.0
    odds_to_continue = 1-odds_to_stop
    fair_at_step = fair
    if int(7-fair) == 3:
        fair_at_step = 5
    elif int(7-fair) == 2:
        fair_at_step = 5.5
    elif int(7-fair) == 1:
        fair_at_step = 6
    return helper(n+1,N,odds_to_stop*fair_at_step + odds_to_continue*fair)


print die_game_fair_value(10000)