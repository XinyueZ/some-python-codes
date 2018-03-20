from countries import country_list

"""
country_counts, whose keys are country names, and whose values are the number of times 
the country occurs in the countries list. Write your code in the app.py file.
"""
country_counts = {}
for country in country_list:
    current_count = country_counts.get(country, 0)
    current_count += 1
    country_counts[country] = current_count


print(country_counts)
