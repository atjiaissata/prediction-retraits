import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

def week_of_month(date):

        """Calcule le numéro de la semaine dans le mois pour une date donnée."""
        first_day_of_month = date.replace(day=1)
        return (date.day - 1) // 7 + 1

def ajout_bruit(data):

    """Cette fonction permet  de rajouter des données fictives dans la base de données
    Paramètres:
    ----------

    data: dataframe
    """
    
    new_dates = pd.date_range(start='2016-01-01', end='2018-12-31', freq='D')

    new_data = pd.DataFrame({'DATE_HISTO': new_dates})

    new_data['INDEX_DAY_WEEK'] = new_data['DATE_HISTO'].dt.weekday

    
    new_data['INDEX_WEEK_MONTH'] = new_data['DATE_HISTO'].apply(week_of_month)

    new_data['INDEX_WEEK_YEAR'] = new_data['DATE_HISTO'].dt.isocalendar().week

    new_data['IS_PENSION_DAY'] = new_data['DATE_HISTO'].dt.day.isin([19, 20, 21]).astype(int)

    new_data['IS_SALARY_DAY'] = new_data['DATE_HISTO'].dt.day.isin([1, 2, 3, 4, 5, 29, 30, 31]).astype(int)

    new_data['IS_WEEKEND_DAY'] = new_data['INDEX_DAY_WEEK'].isin([5, 6]).astype(int)

    new_data['VAL_DD'] = new_data['DATE_HISTO'].dt.day
    new_data['VAL_MM'] = new_data['DATE_HISTO'].dt.month
    new_data['VAL_YYYY'] = new_data['DATE_HISTO'].dt.year

    new_data['REF_ID_ATM'] = 116


    mean_withdrawal = data['MNT_RETRAIT'].mean()
    std_withdrawal = data['MNT_RETRAIT'].std()

    new_data['MNT_RETRAIT'] = np.random.normal(mean_withdrawal, std_withdrawal, len(new_data))
    new_data['MNT_RETRAIT'] = new_data['MNT_RETRAIT'].round(0).astype(int)


    augmented_data = pd.concat([new_data, data], ignore_index=True)

    augmented_data['DATE_HISTO'] = pd.to_datetime(augmented_data['DATE_HISTO'], format='%d/%m/%Y')
    augmented_data = augmented_data.sort_values(by='DATE_HISTO')
    
    augmented_data['MOVING_AVERAGE_3D'] = augmented_data['MNT_RETRAIT'].rolling(window=3, closed='left' ).mean()
    augmented_data['MOVING_STD_7D'] = augmented_data['MNT_RETRAIT'].rolling(window=7, closed='left').std()
    augmented_data['MOVING_AVERAGE_7D'] = augmented_data['MNT_RETRAIT'].rolling(window=7, closed='left').mean()
    augmented_data['SUM_WITHDRAWAL_7D'] = augmented_data['MNT_RETRAIT'].rolling(window=7, closed='left').sum()

    augmented_data['WITHDRAWAL_RATIO_3D'] = augmented_data['MNT_RETRAIT'] / augmented_data['MOVING_AVERAGE_3D']
    augmented_data['WITHDRAWAL_RATIO_7D'] = augmented_data['MNT_RETRAIT'] / augmented_data['MOVING_AVERAGE_7D']
    augmented_data['IS_WITHDRAWAL_SPIKE'] = (augmented_data['MNT_RETRAIT'] > augmented_data['MOVING_AVERAGE_7D'] * 1.5).astype(int)
    
    augmented_data = augmented_data.dropna(subset=["MOVING_STD_7D", "MOVING_AVERAGE_3D"])

    print(augmented_data.isna().sum()/augmented_data.shape[0])

    augmented_data.set_index('DATE_HISTO', inplace=True)
    augmented_data.to_csv("augmented_dataset.csv", index=True)
    print("Nouveau dataset sauvegardé !!!")

    return augmented_data



