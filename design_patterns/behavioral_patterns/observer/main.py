from kpis import KPIs
from current_kpis import CurrentKPIs
from forecast_kpis import ForecastKPIs


if __name__ == '__main__':
    # kpis = KPIs()
    # currentKPIs = CurrentKPIs(kpis)
    # forecastKPIs = ForecastKPIs(kpis)
    with KPIs as kpis:
        with CurrentKPIs(kpis), ForecastKPIs(kpis):
            kpis.set_kpis(25, 10, 5)
            kpis.set_kpis(100, 50, 30)
            kpis.set_kpis(50, 10, 20)

    print(f'\n** Detaching the currentKPIS observer')
    kpis.set_kpis(150, 110, 120)
