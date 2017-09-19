# Enter your code here. Read input from STDIN. Print output to STDOUT
start_year,start_month,end_year,end_month = map(int,raw_input().replace(" ","").replace("-",",").split(','))
raw_input() #skip line
result = {}

def time_in_range(year,month):
    if year not in range(start_year,end_year+1):
        return False
    if year == start_year:
        if month < start_month:
            return False
    if year == end_year:
        if month >= end_month:
            return False
    return True

def complete_month(month):
    if month < 10:
        return '0'+str(month)
    return str(month)

while(True):
    try:
        date,engagement,number = raw_input().replace(" ","").split(",")
        date_year,date_month = map(int,date.split("-")[:-1])
        number = int(number)
        if time_in_range(date_year,date_month):
            if date_year in result:
                if date_month in result[date_year]:
                    if engagement in result[date_year][date_month]:
                        result[date_year][date_month][engagement]+=number
                    else:
                        result[date_year][date_month][engagement]=number
                else:
                    result[date_year][date_month] = {engagement:number}
            else:
                result[date_year] = {date_month:{engagement:number}}
            
    except EOFError:
        break;
     
def output_builder(result):
    str_list = []
    for each_year in sorted(result.keys(),reverse=True):
        for each_month in sorted(result[each_year].keys(),reverse=True): 
            str_ = str(each_year)+'-'+complete_month(each_month)+', '
            for each_engagement in sorted(result[each_year][each_month].keys()):
                str_+=(str(each_engagement)+', '+str(result[each_year][each_month][each_engagement])+', ')
            print str_[:-2]
output_builder(result)
