# ------------------------------------------------- #
#            MAPPING FUNCTIONS                      #
# ------------------------------------------------- #
def Car_age (value):
    return(2022 - value)

def Old_or_New (value):
    if value < 15:
        return('New')
    else:
        return('Old')

def Car_Usage_level(value):
    if value>=5000:
        return('Highly Driven')
    elif 1000<=value<5000:
        return('Moderately Driven')
    else:
        return('Less Driven')

def State_level(value):
    if value>=100000:
        return('High')
    elif 50000<=value<100000:
        return('Moderate')
    else:
        return('Less')

def City_level(value):
    if value>=10000:
        return('High')
    elif 1000<=value<10000:
        return('Moderate')
    else:
        return('Less')
