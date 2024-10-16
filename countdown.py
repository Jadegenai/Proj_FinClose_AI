import pandas as pd
from datetime import datetime
import time
import streamlit as st


def last_business_day():
    # # Get the current date and time
    # curr_timestamp = pd.Timestamp.now()
    # curr_year = curr_timestamp.year
    # curr_month = curr_timestamp.month
    # last_bus_date = pd.Timestamp(curr_year, curr_month, 1) + pd.offsets.MonthEnd(0)
    # # Adjust the date if it falls on a weekend (Saturday=5, Sunday=6)
    # while last_bus_date.weekday() > 4:
    #     last_bus_date -= pd.DateOffset(days=1)
    # date_plus_3_working_days = last_bus_date + pd.offsets.BDay(3)
    # date_minus_3_working_days = last_bus_date - pd.offsets.BDay(3)
    # return date_minus_3_working_days.date(), date_plus_3_working_days.date()
    current_date = pd.Timestamp.now().date()
    date_plus_5_working_days = current_date + pd.offsets.BDay(5)
    return current_date, date_plus_5_working_days.date()


def last_bus_day_countdown():
    first_bus_day, last_bus_day = last_business_day()
    formatted_first_bd = first_bus_day.strftime('%d-%b-%Y')
    formatted_last_bd = last_bus_day.strftime('%d-%b-%Y')
    # Convert current timestamp and last business day to datetime objects
    date_obj = datetime.combine(last_bus_day, datetime.min.time())
    curr_Timestamp = pd.Timestamp.now()
    countDown = date_obj - curr_Timestamp.to_pydatetime()
    cd_days = countDown.days
    cd_total_seconds = countDown.seconds
    cd_hours = cd_total_seconds // 3600
    cd_minutes = (cd_total_seconds % 3600) // 60
    cd_seconds = cd_total_seconds % 60
    return formatted_first_bd, formatted_last_bd, cd_days, cd_hours, cd_minutes, cd_seconds

# st.set_page_config()
# ph = st.empty()
#
# while 1 == 1:
#     formatted_last_bd, remain_days, remain_hours, remain_minutes, remain_seconds = last_bus_day_countdown()
#     ph.metric("Countdown", f"Last Bus Day {formatted_last_bd} {remain_days} Days {remain_hours:02d}:{remain_minutes:02d}:{remain_seconds:02d}")
#     time.sleep(1)
