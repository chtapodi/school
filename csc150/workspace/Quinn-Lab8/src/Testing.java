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
     * the 