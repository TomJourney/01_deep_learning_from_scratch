import datetime

def get_now_year_month_day_hour_minite_second():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")