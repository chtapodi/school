/**
 * This class contains a collection of methods that help with testing.
 * 
 * @author Kristina Striegnitz, Aaron Cass, Chris Fernandes
 * @version 4/27/16
 */
public class Testing {

    private static boolean VERBOSE = false;
    private static int numTests;
    private static int numFails;

    /**
     * Toggles between a lot of output and little output.
     * 
     * @param verbose
     *            If verbose is true, then complete information is printed,
     *            whether the tests passes or fails. If verbose is false, only
     *            failures are printed.
     */
    public static void setVerbose(boolean verbose)
    {
        VERBOSE = verbose;
    }

    /**
     * Each of the assertEquals methods tests whether the actual
     * result equals the expected result. If it does, then the test
     * passes, otherwise it fails.
     * 
     * The only difference between these methods is the types of the
     * parameters.
     *
     * All take a String message and two values of some other type to
     * compare:
     * 
     * @param message
     *            a message or description of the test
     * @param expected
     *            the correct, or expected, value
     * @param actual
     *            the actual value
     */
    public static void assertEquals(String message, boolean expected,
                                    boolean actual)
    {
        printTestCaseInfo(message, "" + expected, "" + actual);
        if (expected == actual) {
            pass();
        } else {
            fail(message);
        }
    }
    
    public static void assertEquals(String message, int expected, int actual)
    {
        printTestCaseInfo(message, "" + expected, "" + actual);
        if (expected == actual) {
            pass();
        } else {
            fail(message);
        }
    }

    public static void assertEquals(String message, Object expected,
                                    Object actual)
    {
        String expectedString = "<<null>>";
        String actualString = "<<null>>";
        if (expected != null) {
            expectedString = expected.toString();
        }
        if (actual != null) {
            actualString = actual.toString();
        }
        printTestCaseInfo(message, expectedString, actualString);

        if (expected == null) {
            if (actual == null) {
                pass();
            } else {
                fail(message);
            }
        } else if (expected.equals(actual)) {
            pass();
        } else {
            fail(message);
        }
    }

    /**
     * Asserts that a given boolean must be true.  The test fails if
     * the boolean is not true.
     * 
     * @param message The test message
     * @param actual The boolean value asserted to be true.
     */
    public static void assertTrue(String message, boolean actual)
    {
        assertEquals(message, true, actual);
    }
    
    /**
     * Asserts that a given boolean must be false. The test fails if
     * the boolean is not false (i.e. if it is true).
     * 
     * @param message The test message
     * @param actual The boolean value asserted to be false.
     */
    public static void assertFalse(String message, boolean actual)
    {
        assertEquals(message, false, actual);
    }

    private static void printTestCaseInfo(String message, String expected,
                                          String actual)
    {
        if (VERBOSE) {
            System.out.println(message + ":");
            System.out.println("expected: " + expected);
            System.out.println("actual:   " + actual);
        }
    }

    private static void pass()
    {
        numTests++;
    	
        if (VERBOSE) {
            System.out.println("--PASS--");
            System.out.println();
        }
    }

    private static void fail(String description)
    {
        numTests++;
        numFails++;

        if (!VERBOSE) {
            System.out.print(description + "  ");
        }
        System.out.println("--FAIL--");
        System.out.println();
    }

    /**
     * Prints a header for a section of tests.
     * 
     * @param sectionTitle The header that should be printed.
     */
    public static void testSection(String sectionTitle)
    {
        if (VERBOSE) {
            int dashCount = sectionTitle.length();
            System.out.println(sectionTitle);
            for (int i = 0; i < dashCount; i++) {
                System.out.print("-");
            }
            System.out.println();
            System.out.println();
        }
    }
    
    /**
     * Initializes the test suite. Should be called before running any
     * tests, so that passes and fails are correctly tallied.
     */
    public static void startTests()
    {
        System.out.println("Starting Tests");
        System.out.println();
        numTests = 0;
        numFails = 0;
    }

    /**
     * Prints out summary data at end of tests.  Should be called
     * after all the tests have run.
     */
    public static void finishTests()
    {
        System.out.println("==============");
        System.out.println("Tests Complete");
        System.out.println("==============");
        int numPasses = numTests - numFails;

        System.out.print(numPasses + "/" + numTests + " PASS ");
        System.out.printf("(pass rate: %.1f%s)\n",
                          100 * ((double) numPasses) / numTests,
                          "%");

        System.out.print(numFails + "/" + numTests + " FAIL ");
        System.out.printf("(fail rate: %.1f%s)\n", 
                          100 * ((double) numFails) / numTests,
                          "%");
    }

}
