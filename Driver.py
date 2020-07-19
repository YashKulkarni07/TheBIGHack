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

    all_countries_S = []
    all_countries_I = []
    all_countries_R = []

    # Extract the us population from the popdataframe
    for i in range (0,len(df)):
        country_name = df.iloc[i][0]
        try:
            country_pop = int(internation_populations.loc[internation_populations['Country'] == country_name]["Population"])
        except:
            country_pop = 1000000



        country_infected_to_date = df.iloc[i][len(df.iloc[i]) -1 ]
        country_death_to_date = df1.iloc[i][len(df1.iloc[i]) - 1]
        model = SIRModel(country_pop,country_infected_to_date,country_death_to_date,t)
        S, I, R = model.run()
        all_countries_S.append(S)
        all_countries_I.append(I)
        all_countries_R.append(R)
    countries = countries = df. iloc[:, 0].tolist()

    # write to data to csv files
    write_file("S.csv", all_countries_S,countries)
    write_file("I.csv", all_countries_I,countries)
    write_file("R.csv", all_countries_R,countries)






def write_file(filename, data, country_names):
    s_out = open(filename,'w')
    s_out.write("Country,")
    for i in range(0,160):
        s_out.write(str(i) + ",")
    s_out.write("\n")
    for a in range(0,len(data)):
        s_out.write(country_names[a] + ",")
        for z in data[a]:
            s_out.write(str(z) + ",")
        s_out.write("\n")
    s_out.close()

if __name__ == "__main__":
    main()