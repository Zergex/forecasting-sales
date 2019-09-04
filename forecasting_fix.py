import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as mp
import pandas as pd
import csv



def holt_winters(params): #метод Холта-Винтерса, возвращает ошибку
    alpha, beta, gamma = params
    s = 12
    finisher = 4  # сколько периодов оставляем с конца для прогноза и вычисления ошибки
    forecasting = 2  # на сколько периодов вперед прогнозируем

    trend_init = 0 # вычисление начального значения тренда
    for i in range(s):
        trend_init += float(series[i + s] - series[i]) / s
    trend_init = trend_init/s
    trend = [trend_init]

    seasonals = {} # вычисление начальных сезонных компонент
    season_averages = []
    n_seasons = int(len(series) / s)
    for j in range(n_seasons):
        season_averages.append(sum(series[s * j:s * j + s]) / float(s))
    for i in range(s):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[s * j + i] - season_averages[j]
        seasonals[i] = sum_of_vals_over_avg / n_seasons

    initial_seasonals = seasonals

    result = []
    seasonals_after = []

    for i in range(len(series) + forecasting):
        if i == 0:  # начальные значения
            smooth = [series[0]]
            result.append(series[0])
            continue
        if i >= (len(series)-finisher):  # прогнозирование
            m = i - len(series) + 1
            if ((smooth[-1] + m * trend[-1]) + seasonals[i % s]) < 0:
                result.append(0)
            else:
                result.append((smooth[-1] + m * trend[-1]) + seasonals[i % s])
        else:
            smooth.append(alpha * (series[i] - seasonals[i % s]) + (1 - alpha) * (smooth[i-1] + trend[i-1]))
            trend.append(beta * (smooth[i] - smooth[i-1]) + (1 - beta) * trend[i-1])
            seasonals[i % s] = gamma * (series[i] - smooth[i]) + (1 - gamma) * seasonals[i % s]
            seasonals_after.append(seasonals[i % s])
            result.append(smooth[i] + trend[i] + seasonals[i % s])

    mp.close()
    fig, subs = mp.subplots(2, 2)

    subs[0, 0].set_xlabel("Номера периодов")
    subs[0, 0].set_ylabel("Объём продаж")
    subs[0, 0].grid(True)
    subs[0, 0].plot(series)
    subs[0, 0].plot(result[0:(len(series) - finisher)], color="red")
    subs[0, 0].plot(range((len(series) - finisher - 1), len(result)), result[(len(series) - finisher - 1):], color="orange")
    subs[0, 0].legend(["Изначальный ряд", "Полученная модель", "Прогноз"])

    subs[1, 0].set_xlabel("Номера периодов")
    subs[1, 0].set_ylabel("Значения компонент")
    subs[1, 0].grid(True)
    subs[1, 0].plot(smooth)
    subs[1, 0].plot(trend)
    subs[1, 0].plot(seasonals_after)
    subs[1, 0].legend(["Уровень", "Тренд", "Сезонные компоненты"])

    subs[0, 1].set_xlabel("Номера периодов")
    subs[0, 1].set_ylabel("Объём продаж")
    subs[0, 1].grid(True)
    subs[0, 1].plot(series)
    subs[0, 1].plot(smooth)
    subs[0, 1].legend(["Изначальный ряд", "Уровень"])

    subs[1, 1].set_xlabel("Номера периодов")
    subs[1, 1].set_ylabel("Значение компоненты")
    subs[1, 1].grid(True)
    subs[1, 1].plot(initial_seasonals.values())
    subs[1, 1].legend(["Начальные сезонные компоненты"])

    global res_forplot
    res_forplot=result[0:(len(series) - finisher)]
    global forc_forplot
    forc_forplot=result[(len(series) - finisher - 1):len(series)]
    global forc_forplot_x
    forc_forplot_x=range((len(series) - finisher - 1), len(series))



    deviation_sum = 0

    if finisher==0: #вычисление ошибки
        for i in range(0, len(series)):
            series_i = series[i]
            if series_i==0:
                series_i=1
            deviation_sum += ((series_i - result[i]) ** 2 / series_i ** 2)
        precision = deviation_sum / len(series)
    else:
        for i in range(len(series) - finisher, len(series)):
            series_i = series[i]
            if series_i == 0:
                series_i = 1
            deviation_sum += ((series_i - result[i]) ** 2 / series_i ** 2)
        precision = deviation_sum / finisher

    return precision

################################################

def holt_winters_short(params): #метод Холта-Винтерса, возвращает результат прогноза
    alpha, beta, gamma = params
    s = 12
    finisher = 0
    forecasting = 2

    trend_init = 0
    for i in range(s):
        trend_init += float(series[i + s] - series[i]) / s
    trend_init = trend_init/s
    trend = [trend_init]

    seasonals = {}
    season_averages = []
    n_seasons = int(len(series) / s)
    for j in range(n_seasons):
        season_averages.append(sum(series[s * j:s * j + s]) / float(s))
    for i in range(s):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[s * j + i] - season_averages[j]
        seasonals[i] = sum_of_vals_over_avg / n_seasons

    initial_seasonals = seasonals

    result = []
    seasonals_after = []

    for i in range(len(series) + forecasting):
        if i == 0:
            smooth = [series[0]]
            result.append(series[0])
            continue
        if i >= (len(series)-finisher):
            m = i - len(series) + 1
            if ((smooth[-1] + m * trend[-1]) + seasonals[i % s])<0:
                result.append(0)
            else:
                result.append(round((smooth[-1] + m * trend[-1]) + seasonals[i % s]))
        else:
            smooth.append(alpha * (series[i] - seasonals[i % s]) + (1 - alpha) * (smooth[i-1] + trend[i-1]))
            trend.append(beta * (smooth[i] - smooth[i-1]) + (1 - beta) * trend[i-1])
            seasonals[i % s] = gamma * (series[i] - smooth[i]) + (1 - gamma) * seasonals[i % s]
            seasonals_after.append(seasonals[i % s])
            result.append(smooth[i] + trend[i] + seasonals[i % s])

    mp.close()

    fig, subs = mp.subplots(2, 2)

    subs[0, 0].set_xlabel("Номера периодов")
    subs[0, 0].set_ylabel("Объём продаж")
    subs[0, 0].grid(True)
    subs[0, 0].plot(series)
    subs[0, 0].plot(result[0:(len(series) - finisher)], color="red")
    subs[0, 0].plot(range((len(series) - finisher - 1), len(result)), result[(len(series) - finisher - 1):], color="orange")
    subs[0, 0].legend(["Изначальный ряд", "Полученная модель", "Прогноз"])

    subs[1, 0].set_xlabel("Номера периодов")
    subs[1, 0].set_ylabel("Значения компонент")
    subs[1, 0].grid(True)
    subs[1, 0].plot(smooth)
    subs[1, 0].plot(trend)
    subs[1, 0].plot(seasonals_after)
    subs[1, 0].legend(["Уровень", "Тренд", "Сезонные компоненты"])

    subs[0, 1].set_xlabel("Номера периодов")
    subs[0, 1].set_ylabel("Объём продаж")
    subs[0, 1].grid(True)
    subs[0, 1].plot(series)
    subs[0, 1].plot(smooth)
    subs[0, 1].legend(["Изначальный ряд", "Уровень"])

    subs[1, 1].set_xlabel("Номера периодов")
    subs[1, 1].set_ylabel("Значение компоненты")
    subs[1, 1].grid(True)
    subs[1, 1].plot(initial_seasonals.values())
    subs[1, 1].legend(["Начальные сезонные компоненты"])

    global res_forplot
    res_forplot = result[0:(len(series) - finisher)]
    global forc_forplot
    forc_forplot = result[(len(series) - finisher - 1):]
    global forc_forplot_x
    forc_forplot_x = range((len(series) - finisher - 1), len(result))

    deviation_sum = 0

    if finisher==0:
        for i in range(0, len(series)):
            series_i = series[i]
            if series_i==0:
                series_i=1
            deviation_sum += ((series_i - result[i]) ** 2 / series_i ** 2)
        precision = deviation_sum / len(series)
    else:
        for i in range(len(series) - finisher, len(series)):
            series_i = series[i]
            if series_i == 0:
                series_i = 1
            deviation_sum += ((series_i - result[i]) ** 2 / series_i ** 2)
        precision = deviation_sum / finisher

    return result

###############################################

sheet = pd.read_excel('продажа товара по месяцам 2019 2.xlsx')

data = (((sheet.fillna(0)).iloc[1:, 0:]).as_matrix()).tolist()
data2 = (((sheet.fillna(0)).iloc[1:, 0:]).as_matrix()).tolist()

res_forplot = []
forc_forplot = []
forc_forplot_x = []
forc2_forplot = []

titles_list=[]
for i in data:
    titles_list.append(i[0])

print(titles_list.index(data[3][0]))

zeros = [0]*24

cat_numbers = []
line_number = 0
for i in data:
    if i[0][-1]==",":
        cat_numbers.append(line_number)
    line_number += 1
cat_numbers.append(len(data))

for i in range(11, (len(cat_numbers)-1)):
    test_data = data[cat_numbers[i]+1:cat_numbers[i+1]]

    kolvo_iter = 1
    for j in test_data:
        j.append(sum(j[1:]))

    all_sales = sum([item[-1] for item in test_data])

    test_data = sorted(test_data, key=lambda x: x[-1], reverse=True)
    percentage = []

    narast = 0
    for j in test_data:
        j.append(j[-1] * 100 / all_sales)
        narast += j[-1]
        j.append(narast)
        percentage.append(narast)
        j.append((kolvo_iter / len(test_data)) * 100)
        kolvo_iter += 1
        j.append(narast + j[-1])

    test_data = sorted(test_data, key=lambda x: x[-1], reverse=False)

    for j in test_data:
        if (j[-1] <= 100):
            j.append("a")
        elif (j[-1] > 100 and j[-1] <= 145):
            j.append("b")
        elif (j[-1] > 145):
            j.append("c")

    for j in test_data:
        aver = sum(j[1:-6]) / len(j[1:-6])
        vsum = 0
        if aver == 0:
            j.append(0)
        else:
            for o in j[1:-6]:
                vsum = vsum + (o - aver) ** 2
            j.append((((vsum / len(j[1:-6])) ** 0.5) / aver) * 100)

    category_ax = []
    category_bx = []
    category_cx = []
    category_ay = []
    category_by = []
    category_cy = []
    category_az = []
    category_bz = []
    category_cz = []

    category_ax_n = []
    category_bx_n = []
    category_cx_n = []
    category_ay_n = []
    category_by_n = []
    category_cy_n = []
    category_az_n = []
    category_bz_n = []
    category_cz_n = []

    a_count = 0
    b_count = 0
    c_count = 0

    for j in test_data:
        if j[-2] == "a":
            if j[-1]<=10:
                category_ax.append(j[0:-7])
                category_ax_n.append(j[1:-7])
            if j[-1]>10 and j[-1]<=25:
                category_ay.append(j[0:-7])
                category_ay_n.append(j[1:-7])
            if j[-1]>25:
                category_az.append(j[0:-7])
                category_az_n.append(j[1:-7])
            a_count += 1
        if j[-2] == "b":
            if j[-1] <= 10:
                category_bx.append(j[0:-7])
                category_bx_n.append(j[1:-7])
            if j[-1] > 10 and j[-1] <= 25:
                category_by.append(j[0:-7])
                category_by_n.append(j[1:-7])
            if j[-1] > 25 :
                category_bz.append(j[0:-7])
                category_bz_n.append(j[1:-7])
            b_count += 1
        if j[-2] == "c":
            if j[-1] <= 10:
                category_cx.append(j[0:-7])
                category_cx_n.append(j[1:-7])
            if j[-1] > 10 and j[-1] <= 25:
                category_cy.append(j[0:-7])
                category_cy_n.append(j[1:-7])
            if j[-1] > 25:
                category_cz.append(j[0:-7])
                category_cz_n.append(j[1:-7])
            c_count += 1


    #mp.grid(True)
    #mp.xlabel("Номера товаров")
    #mp.ylabel("Доля объема продаж, %")
    #print(a_count, b_count, c_count)
    #mp.plot(range(0, a_count), percentage[0:a_count])
    #mp.plot(range(a_count, a_count + b_count), percentage[a_count:a_count + b_count])
    #mp.plot(range(a_count + b_count, len(test_data)), percentage[a_count + b_count:len(test_data)])
    #mp.legend(["Товары класса А", "Товары класса B", "Товары класса С"])
    #mp.show()

    sum_ax = [sum(x) for x in zip(*category_ax_n)]
    sum_bx = [sum(x) for x in zip(*category_bx_n)]
    sum_cx = [sum(x) for x in zip(*category_cx_n)]

    sum_ay = [sum(x) for x in zip(*category_ay_n)]
    sum_by = [sum(x) for x in zip(*category_by_n)]
    sum_cy = [sum(x) for x in zip(*category_cy_n)]

    sum_az = [sum(x) for x in zip(*category_az_n)]
    sum_bz = [sum(x) for x in zip(*category_bz_n)]
    sum_cz = [sum(x) for x in zip(*category_cz_n)]

    cats = [sum_ax, sum_bx, sum_cx, sum_ay, sum_by, sum_cy, sum_az, sum_bz, sum_cz]
    categories = [category_ax, category_bx, category_cx, category_ay, category_by, category_cy, category_az, category_bz, category_cz]
    categories_n = [category_ax_n, category_bx_n, category_cx_n, category_ay_n, category_by_n, category_cy_n, category_az_n, category_bz_n, category_cz_n]
    cat_titles = ["Категория АХ", "Категория ВХ", "Категория СХ", "Категория AY", "Категория BY", "Категория CY", "Категория AZ", "Категория BZ", "Категория CZ"]
    for k in range(0, len(categories)):
        forecast_data = []
        series = cats[k]
        if series==[]:
            continue
        else:
            print(cat_titles[k], "- размер:", len(categories[k]))
        initial_guess = [0.5, 0.5, 0.5]  # начальные значения
        bnds = ((0, 1), (0, 1), (0, 1))  # ограничения на параметры

        result = opt.minimize(holt_winters, initial_guess,
                              bounds=bnds)  # минимизация. На вход подается функция, начальные значения и ограничения на параметры.

        fitted_params = result.x  # здесь подобранные в результате оптимизации параметры
        optimized_precision = 1-holt_winters(fitted_params)
        print("Полученные коэффициенты: ", fitted_params, "Точность: ", optimized_precision*100, "%")
        mp.show()
        #mp.close()
        #mp.grid(True)
        #mp.xlabel("Номера периодов", fontsize=23)
        #mp.ylabel("Объем продаж", fontsize=23)
        #mp.plot(series)
        #mp.plot(res_forplot, color="red")
        #mp.plot(forc_forplot_x, forc_forplot, color="orange")
        #mp.legend(["Исходный ряд", "Полученная модель", "Прогноз"], fontsize=20)
        #mp.show()
        holt_winters_short(fitted_params)
        #mp.close()
        #mp.grid(True)
        #mp.xlabel("Номера периодов", fontsize=23)
        #mp.ylabel("Объем продаж", fontsize=23)
        #mp.plot(series)
        #mp.plot(res_forplot, color="red")
        #mp.plot(forc_forplot_x, forc_forplot, color="orange")
        #mp.legend(["Исходный ряд", "Полученная модель", "Прогноз"], fontsize=20)
        mp.show()
        s_index = 0
        for series in categories_n[k]:
            if series[0:24] == zeros or series[-24:] == zeros:
                s_index += 1
                continue
            fdata=(holt_winters_short(fitted_params)[-2:])
            print(categories[k][s_index][0])
            t_index = titles_list.index(categories[k][s_index][0])
            data2[t_index].extend(fdata)
            print(data2[t_index])
            s_index += 1

            mp.show()

with open('прогноз.csv', 'w', newline='', encoding='utf-8') as f:
    fc = csv.writer(f)
    fc.writerows(data2)