def is_in_list(list, expect):
	"""
	Check whether expect is in list or not.
	Returns when finds expect.
	"""
	for element in list:
		if element == expect: return True
	return False

def remove_duplicates(list):
	"""
	Takes a list as its argument and returns a new list containing the unique 
	elements of the original list. The elements in the new list without duplicates 
	can be in any order.
	"""
	new_list = []
	for element in list:
		if not is_in_list(new_list, element):
			new_list.append(element)
	return new_list

print(remove_duplicates(['Angola', 'Maldives', 'India', 'United States']) == ['Angola', 'Maldives', 'India', 'United States'])
print(remove_duplicates(['Angola', 'Maldives', 'India', 'United States', 'India']) == ['Angola', 'Maldives', 'India', 'United States'])
