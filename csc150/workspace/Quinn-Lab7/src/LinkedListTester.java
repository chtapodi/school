/**
 * Tests new (and old?) LinkedList methods.
 * 
 * @author <em>Xavier QUinn</em>, Matt Anderson, Chris Fernandes, and Aaron
 *         Cass.
 * @version 02/16/17
 */
public class LinkedListTester {
	public static void main(String[] args) {

		Testing.startTests();
		Testing.setVerbose(true);

		test231();
		// make your own test methods and call them here.

		Testing.finishTests();

	}

	/**
	 * Adds events in the order 1, 2, 3 that should end up in the order 2, 3, 1.
	 */
	public static void test231() {
		LinkedList list = new LinkedList();

		Event e1 = new Event("chess", 2009, 1, 25, 1900, 1930);
		Event e2 = new Event("boy scouts", 2009, 1, 23, 900, 1000);
		Event e3 = new Event("book club", 2009, 1, 25, 800, 830);
		Event e4 = new Event("girl scouts", 2009, 1, 23, 900, 1000);
		Event e5 = new Event("Cretaceous–Paleogene extinction event", -65997983, 0, 0, 0, 12000);
		Event e6 = new Event(null, 0, 0, 0, 0, 0);


		
		
		
		

		
		list.insertSorted(e1);
		list.insertSorted(e2);
		list.insertSorted(e3);
		list.insertSorted(e4);
		list.insertSorted(e5);
		list.insertSorted(e6);
		

		Event[] expected = new Event[] { e5, e6, e2, e4, e3, e1 };

		testArrayAndList("Inserts in sorted order", expected, list);
	}

	// Some private helper functions.
	private static void testArrayAndList(String message, Event[] expected,
			LinkedList list) {
		Testing.assertEquals(message, arrayToString(expected), list.toString());
	}

	private static String arrayToString(Object[] array) {
		String result = "(";
		int size = array.length;

		for (int i = 0; i < size - 1; i++) {
			result += array[i].toString() + ",\n";
		}
		result += array[size - 1] + ")";

		return result;
	}
}