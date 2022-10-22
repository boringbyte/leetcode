# Pytest
Pytest is a testing framework. 

    - Open Source
    - Parallel running of test cases
    - Fixtures
    - Parametrization
    - Pre-conditions before starting a test
    - Post-conditions after ending a test
    - Skip
    - Fail

## Main concepts:
- All the test files start with test_ prefix or end with _test.py
- All the test functions start with test_ prefix or end with _test.
- You can run test by running command **pytest** in the directory where the test python file is there.
- __.__ after the test_first.py after running a test case indicates that test case passed successfully.
- Number of __.__'s after the test_first.py after running pytest command indicates, how many test passed. If there are any failures, then we see __F__.
- pytest -rA means all the test cases. We see a summary of the test cases.
- pytest will run all files from current directory or subdirectory test_*.py or *_test.py
- pytest -h or --help for getting help
- pytest --markers --> to know more details about markers
- pytest --fixtures --> to know more details about fixtures
- pytest test_second -k login --> run test cases which has keyword "login"
- pytest -k login --> run test cases which has keyword "login" in the current directory
- pytest -v or --verbose --> means verbose
- pytest -rA --junitxml="Report1.xml" --> dumps results to xml file
- pytest --html=HTMLReport.html --> dumps results to html file.
- By using markers, we are providing additional information to the test function.
- pytest -v -m webtest if there is a test with maker @pytest.mark.webtest. webtest is our own marker.
  - pytest test_markers_demo.py -m smoke
  - pytest test_markers_demo.py -m regression
  - pytest test_markers_demo.py -m "smoke or regression"
  - We need to add markers to pytest.ini file.
  - Add "addopts = -rA" and other info to pytest.ini file to add more information to ini file.
- @pytest.mark.skip --> unconditional skip
- @pytest.mark.skipif(sys.version_info.minor < 10, reason='Current support version is >= 10') --> conditional skip
- xfail means, we expect the test to fail for some reason.
- parametrize decorator enables parametrization of arguments for a test function so that we can pass multiple test args using the same test function.
  - n named parameters can be passed and their values are passed in the form of a list
  - like @pytest.mark.parametrize("test_input, expected", [("3+5", 8), ("2+4", 6)])
- Fixtures
  - Preconditions like setup, connection to DB or API
  - Post condition like clean up, close etc. it is placed after yield.
  - User global variables inside test functions to access variables created inside fixture.
- MonkeyPatch is another important topic to study.
- pytest-xdist library is used to parallelize tests.
  - pytest test_fixtures -n 3