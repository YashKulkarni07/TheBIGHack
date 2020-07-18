import pandas as pd
import numpy as np
from BetterSIRModel import SIRModel

def main():
    # Load in the dataframes
    df = pd.read_csv("./Data-20200718T211646Z-001/Data/International/International_covid_cases_data.csv", index_col=0)
    df1 = pd.read_csv("./Data-20200718T211646Z-001/Data/International/International_covid_deaths_data.csv", index_col=0)
    df2 = pd.read_csv("./Data-20200718T211646Z-001/Data/LA/LA_cities_covid_data.csv", index_col=0)
    df3 = pd.read_csv("./Data-20200718T211646Z-001/Data/US Counties/US_county_covid_cases_data.csv", index_col=0)
    df4 = pd.read_csv("./Data-20200718T211646Z-001/Data/US Counties/US_county_covid_deaths_data.csv", index_col=0)


    # A grid of time points (in days):
    t = np.linspace(0, 160, 160)
    model = SIRModel(1000,1,0,t)
    S, I, R = model.run()
    model.plot(S, I, R)

if __name__ == "__main__":
    main()