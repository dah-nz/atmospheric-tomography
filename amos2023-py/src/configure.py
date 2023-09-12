from parameters import test

filename = "./src/parameters/settings.json"

test.configure(filename)

settings = test.load(filename)

print(settings)