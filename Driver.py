import pandas as pd
import numpy as np
from BetterSIRModel import SIRModel

def main():
    # Load in the dataframes
    df = pd.read_csv("./Data-20200718T211646Z-001/Data/International/International_covid_cases_data.csv")
    df1 = pd.read_csv("./Data-20200718T211646Z-001/Data/International/International_covid_deaths_data.csv")
    df2 = pd.read_csv("./Data-20200718T211646Z-001/Data/LA/LA_cities_covid_data.csv", index_col=0)
    df3 = pd.read_csv("./Data-20200718T211646Z-001/Data/US Counties/US_county_covid_cases_data.csv", index_col=0)
    df4 = pd.read_csv("./Data-20200718T211646Z-001/Data/US Counties/US_county_covid_deaths_data.csv", index_col=0)
    internation_populations = pd.read_csv("./Data-20200718T211646Z-001/Data/International/population_by_country_2020.csv")

    # A grid of time points (in days):
    t = np.linspace(0, 160,160 )

    # Extract the us population from the popdataframe
    for i in range (0,len(df)):
        country_name = df.iloc[i][0]
        country_pop = int(internation_populations.loc[internation_populations['Country'] == country_name]["Population"])
        country_infected_to_date = df.iloc[i][len(df.iloc[i]) -1 ]
        country_death_to_date = df1.iloc[i][len(df1.iloc[i]) - 1]
        model = SIRModel(country_pop,country_infected_to_date,country_death_to_date,t)
        S, I, R = model.run()
        #model.plot(S, I, R)

if __name__ == "__main__":
    main()