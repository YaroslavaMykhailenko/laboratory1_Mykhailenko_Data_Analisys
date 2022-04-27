from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from scipy import stats
from numpy import mean, square
from scipy.stats import f

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
import_data = pd.read_csv('A12.txt', header=None)
import_data.columns = [f'{i}' for i in range(12)]
print(import_data)

# table_format = PrettyTable()
# table_format.add_column('Фактор', ['Parametr'])
# for column_name, column_data in import_data.iteritems():
#     # print(column_data.to_string(index=False))

#     table_format.add_column(f'A{str(int(column_name)+1)}', [round(column_data,2).to_string(index=False)])

# print(table_format)


def ParamGraph(df):
    fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(27, 10))
    ax = ax.flatten()

    for column_name, column_data in import_data.iteritems():
        print(column_name)
        ax[int(column_name)].plot(column_data, color='red')
        ax[int(column_name)].set_ylabel(f'A{int(column_name) + 1}')

    fig.tight_layout()
    plt.show()


def RateParams(df):
    fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(14, 15))
    ax = ax.flatten()

    for column_name, column_data in import_data.iteritems():
        print('➊------------------➊')
        print(f' Заданий параметр:{int(column_name) + 1}')
        print('➊------------------➊')

        mean_value = column_data.mean()
        dispersion_value = column_data.std()
        mode_value = column_data.mode()[0]
        median_value = column_data.median()
        skew_value = column_data.skew()
        kurtosis_value = column_data.kurtosis()
        shapiro_test = stats.shapiro(column_data)[1]

        print("• Середнє знач параметру:", round(mean_value, 3))
        print("• Дисперсія параметру:", round(dispersion_value, 3))
        print("• Мода параметру:", round(mode_value, 3))
        print("• Медіана параметру:", round(median_value, 3))
        print("• Коефіцієнт асиметрії параметру:", round(skew_value, 3))
        print("• Rоефіцієнт ексцесу параметру:", round(kurtosis_value, 3))

        if shapiro_test > 1e-5:
            print("Розподіл нормальний")
            
        else:
            print("Розподіл не є нормальний")

        column_data.hist(ax=ax[int(column_name)], bins=25, color='green')
        ax1 = column_data.plot.kde(ax=ax[int(column_name)], secondary_y=True, color='pink')
        ax[int(column_name)].set(title=f'A{int(column_name) + 1}', ylabel = ' ')
        

    fig.tight_layout()
    plt.show()


def OneFactorDispersionAnalysis(df):
    N = import_data.shape[0]
    k = import_data.shape[1]
    si_square_lst = []
    x_i = []
    alfa_g = 0.153
    alpha = 0.95
    double_sum = 0
    square_double_sum = 0
    summa_square = 0 


    table_format = PrettyTable()

    table_format.add_column('Фактор', ['Si Square Value',])


    for i in import_data.columns:
        si_square = (1 / (N - 1) * (pow(import_data[i], 2).agg('sum') - (1 / N) * pow(import_data[i].agg('sum'), 2)))
        si_square_lst.append(si_square)

        table_format.add_column(f'A{str(int(i)+1)}', [round(si_square,2)])

    print(table_format)

    g = max(si_square_lst) / sum(si_square_lst)

    # Перевіримо критерій порівняння
    if g >= alfa_g:

        print("Нульова гіпотеза про рівність дисперсій відхиляється")
    else:
        print("Нульова гіпотеза про рівність дисперсій приймається")

    print("Розглянемо випадок коли гіпотеза приймається:\n\n")

    
    for i in import_data.columns:
        double_sum += pow(import_data[i], 2).agg('sum')
        square_double_sum += pow(import_data[i].agg('sum'), 2)
        summa_square += import_data[i].agg('sum')

        mean_value = import_data[i].agg('mean')
        x_i.append(mean_value)

    s_zero_square = (1 / (k * (N - 1)) * (double_sum - (1 / N) * square_double_sum))
    s_square = (double_sum - (pow(summa_square, 2)) / (k * N)) / (k * N - 1)
    x_double_line = mean(x_i)
    s_a_square = ((N *(pow(x_i - x_double_line, 2).sum())) / (k - 1))



    print("• Оцінка дисперсії, що характеризує розсіювання поза впливом фактора: ", round(s_zero_square, 3))
    print("• Вибіркова дисперсія всіх спостережень: ", round(s_square, 3))
    print("• Оцінка дисперсії, що характеризує зміни параметра, пов'язані з фактором: ", round(s_a_square, 3), '\n\n')

# Оцінка впливу фактора на зміни середнього значення визначається відношенням (вплив значущий з ймовірністю 1-α)

    if (s_a_square / s_zero_square) > f.ppf(alpha, (k - 1), k * (N - 1)):
        print("Вплив фактора значущий")
    else:
        print("Вплив фактора незначущий")





def TwoFactorDispersionAnalysis(df):

    N = 1000
    M = 5
    k = import_data.shape[1]
    alpha = 0.95

    df_double_factor = import_data.groupby(import_data.index // N).agg(list)
    mean_df_double_factor = df_double_factor.applymap(mean)
    squared_sum_df_double_factor = df_double_factor.applymap(square).applymap(sum)


    Q1 = pow(mean_df_double_factor.values,2).sum()
    Q2 = pow(mean_df_double_factor.sum(axis=0), 2).sum() / M
    Q3 = pow(mean_df_double_factor.sum(axis=1), 2).sum() / k
    Q4 = pow(mean_df_double_factor.sum(axis=1).sum(), 2) / (M * k)
    s_zero_square = (Q1 + Q4 - Q2 - Q3) / ((k - 1) * (M - 1))
    s_a_square = (Q2 - Q4) / (k - 1)
    s_b_square = (Q3 - Q4) / (M - 1)

    print("• Q1 буде дорівнювати: ", round(Q1, 3))

    print("• Q2 буде дорівнювати: ", round(Q2, 3))

    print("• Q3 буде дорівнювати: ", round(Q3, 3))

    print("• Q4 буде дорівнювати: ", round(Q4, 3))

    print("\n\nЗнаходимо оцінки дисперсій:")
    print("• S_0_squared дорівнює: ",  round(s_zero_square, 3))

    print("• S_a_squared дорівнює: ", round(s_a_square, 3))

    print("• S_b_squared дорівнює: ", round(s_b_square, 3))



    if (s_a_square / s_zero_square) > f.ppf(alpha, (k - 1), (k - 1) * (M - 1)):
        print("Вплив фактора A значущий")
    else:
        print("Вплив фактора A незначущий")

    if (s_b_square / s_zero_square) > f.ppf(alpha, (M - 1), (k - 1) * (M - 1)):
        print("Вплив фактора B значущий")
    else:
        print("Вплив фактора B незначущий")


    if (s_a_square / s_zero_square) > f.ppf(alpha, (k - 1), (k - 1) * (M - 1)) and (s_b_square / s_zero_square) > f.ppf(alpha, (M - 1), (k - 1) * (M - 1)):
        q5 = squared_sum_df_double_factor.to_numpy().sum()
        s_ab_square = (q5 - N * Q1) / (M * k * (N - 1))

        print("Q5 буде дорівнювати: ",  q5)
        
        print("• S_ab_squared дорівнює: ",  s_ab_square)
    
        if (N * (s_zero_square) / s_ab_square) > f.ppf(alpha, (k - 1) * (M - 1), M * k * (N - 1)):
            print("Вплив факторів значущий ")
        else:
            print("Вплив факторів не значущий ")

    






def start():
    print("\n\nChoose the task you want to solve: \n\n1 - Побудова графіку для кожного параметру\n2 - Для кожного параметру оцінити основні статистичні параметри\n3 - Однофакторний аналіз\n4 - Двофакторний аналіз")
    number = int(input(""))
    if number == 1:
        ParamGraph(import_data)
        again()
    elif number == 2:
        RateParams(import_data)
        again()
    elif number == 3:
        OneFactorDispersionAnalysis(import_data)
        again()
    elif number == 4:
        TwoFactorDispersionAnalysis(import_data)
        again()
    else:
        print("Невірний символ!")


def again():
    print("\n\nБажаєте запустити програму ще раз?:\n1 - так\n2 - ні")
    choice = int(input(""))
    if choice == 1:
        start()

    elif choice == 2:
        return 0


start()