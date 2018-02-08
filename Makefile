# Just so I remember how to run tests
# Requires installation of pytest-cov
test:
	py.test --cov-config .coveragerc --cov-report term-missing --cov=cross_validation cross_validation/tests/
