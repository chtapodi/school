/**
 * Testing suite for InfixPostFix converter
 * 
 * @author Xavier Qunn, Chris Fernandes, and Matt Anderson *I affirm that I have
 *         carried out the attached academic endeavors with full academic
 *         honesty, in accordance with the Union College Honor Code and the
 *         course syllabus.
 */
public class ProjectTesting {

	public static final boolean VERBOSE = true;

	/*
	 * Runs a bunch of tests for the BetterBag class.
	 * 
	 * @param args is ignored
	 */
	public static void main(String[] args) {

		Testing.setVerbose(true);
		Testing.startTests();

		testInserts();

		testRemove();

		testToStringAndPush();

		testPopAndPeek();

		testSizeAndIsEmpty();
		
		ConverterTester();

		Testing.finishTests();

	}

	private static void testInserts() {
		Testing.testSection("Tests insertAtHead, insertAtTail, and toString");
		LinkedList<String> list = new LinkedList<String>();

		LinkedList<String> list2 = new LinkedList<String>();

		LinkedList<Integer> intList = new LinkedList<Integer>();

		list.insertAt(0, "One");
		Testing.assertEquals("Tests addition in empty list at start", "(One)",
				list.toString());
		Testing.assertEquals("Tests addition in empty list capacity", 1,
				list.getLength());

		list.insertAt(5, "Two");
		Testing.assertEquals("Tests addition at location longer than length",
				"(One, Two)", list.toString());
		Testing.assertEquals("Tests addition in empty list capacity", 2,
				list.getLength());

		list.insertAt(1, "Three");
		Testing.assertEquals("Tests addition between nodes",
				"(One, Three, Two)", list.toString());
		Testing.assertEquals("Tests addition in empty list capacity", 3,
				list.getLength());

		list.insertAt(0, "Four");
		Testing.assertEquals("Tests addition at start",
				"(Four, One, Three, Two)", list.toString());
		Testing.assertEquals("Tests addition in empty list capacity", 4,
				list.getLength());

		list.insertAt(-6, "Five");
		Testing.assertEquals("Tests addition at negative index",
				"(Five, Four, One, Three, Two)", list.toString());

		Testing.assertEquals("Tests addition in empty list capacity", 5,
				list.getLength());
		list.insertAt(6, "Six");
		Testing.assertEquals("Tests addition at end",
				"(Five, Four, One, Three, Two, Six)", list.toString());
		Testing.assertEquals("Tests addition in empty list capacity", 6,
				list.getLength());

		list2.insertAt(0, "a");
		list2.insertAt(1, null);
		list2.insertAt(2, "b");

		Testing.assertEquals("Tests addition between nodes", "(a, null, b)",
				list2.toString());

		intList.insertAt(0, 1);
		Testing.assertEquals("Tests addition in empty list at start", "(1)",
				intList.toString());
		Testing.assertEquals("Tests addition in empty list capacity", 1,
				intList.getLength());

	}

	private static void testRemove() {
		Testing.testSection("Tests insertAtHead, insertAtTail, and toString");
		LinkedList<String> list = new LinkedList<String>();
		list.insertAt(10, "One");
		list.insertAt(10, "Two");
		list.insertAt(10, "Three");
		list.insertAt(10, "Four");
		Testing.assertEquals("Just checking", "(One, Two, Three, Four)",
				list.toString());
		Testing.assertEquals("Tests addition in empty list capacity", 4,
				list.getLength());

		Testing.assertEquals("Test removal of last", "Four", list.removeAt(3));
		Testing.assertEquals("Test removal of last", "(One, Two, Three)",
				list.toString());
		Testing.assertEquals("Tests capacity after removal", 3,
				list.getLength());

		Testing.assertEquals("Test removal of first", "One", list.removeAt(0));
		Testing.assertEquals("Test removal of first", "(Two, Three)",
				list.toString());
		Testing.assertEquals("Tests capacity after removal", 2,
				list.getLength());

		Testing.assertEquals("Test removal of first", null, list.removeAt(-5));
		Testing.assertEquals("Test removal of first", "(Two, Three)",
				list.toString());
		Testing.assertEquals("Tests capacity after removal", 2,
				list.getLength());

		Testing.assertEquals("Test removal of first", null, list.removeAt(5));
		Testing.assertEquals("Test removal of first", "(Two, Three)",
				list.toString());
		Testing.assertEquals("Tests capacity after removal", 2,
				list.getLength());
	}

	private static void testToStringAndPush() {
		Testing.testSection("Testing toString and push");

		Stack<String> stack = new Stack<String>();
		Testing.assertEquals(
				"An empty stack. (> indicates the top of the stack)", "{>}",
				stack.toString());

		stack.push("A");
		Testing.assertEquals("A stack with one item", "{>A}", stack.toString());

		stack.push("B");
		stack.push("C");
		Testing.assertEquals("A stack with several items", "{>C, B, A}",
				stack.toString());
	}

	private static void testPopAndPeek() {
		Testing.testSection("Testing Pop and Peak");

		Stack<String> stack = new Stack<String>();
		Testing.assertEquals("An empty stack, poping should return null", null,
				stack.pop());
		Testing.assertEquals("An empty stack, peeking should return null",
				null, stack.peek());

		stack.push("A");

		Testing.assertEquals("Peek a stack with one item ", "A", stack.peek());
		Testing.assertEquals("Pop a stack with one item ", "A", stack.pop());
		Testing.assertEquals("An empty stack, poping should return null", null,
				stack.pop());
		Testing.assertEquals("An empty stack, peeking should return null",
				null, stack.peek());

		stack.push("A");
		stack.push("B");
		stack.push("C");

		Testing.assertEquals("Peek a stack with several items ", "C",
				stack.peek());
		Testing.assertEquals("Peek a stack with several items ", "C",
				stack.pop());

		Testing.assertEquals("Peek a stack with several items ", "B",
				stack.peek());
		Testing.assertEquals("Peek a stack with several items ", "B",
				stack.pop());

		Testing.assertEquals("Peek a stack with several items ", "A",
				stack.peek());
		Testing.assertEquals("Peek a stack with several items ", "A",
				stack.pop());

		Testing.assertEquals("Peek a stack with several items ", null,
				stack.peek());
		Testing.assertEquals("Peek a stack with several items ", null,
				stack.pop());

	}

	private static void testSizeAndIsEmpty() {
		Testing.testSection("Testing toString and push");

		Stack<String> stack = new Stack<String>();
		Testing.assertEquals("An empty stack size", 0, stack.size());
		Testing.assertEquals("An empty stack is empty", true, stack.isEmpty());

		stack.push("A");
		Testing.assertEquals("A stack with one item size", 1, stack.size());
		Testing.assertEquals("A stack with one item is empty", false,
				stack.isEmpty());

		stack.push("B");
		Testing.assertEquals("A stack with two item size", 2, stack.size());
		Testing.assertEquals("A stack with two item is empty", false,
				stack.isEmpty());

		stack.push("C");
		Testing.assertEquals("A stack with three item size", 3, stack.size());
		Testing.assertEquals("A stack with three item is empty", false,
				stack.isEmpty());

		stack.pop();
		Testing.assertEquals("A stack with 2 item size after pop", 2,
				stack.size());
		Testing.assertEquals("A stack with 2 item is empty after pop", false,
				stack.isEmpty());

		stack.pop();
		Testing.assertEquals("A stack with one item size after pop", 1,
				stack.size());
		Testing.assertEquals("A stack with one item is empty after pop", false,
				stack.isEmpty());

		stack.pop();
		Testing.assertEquals("A stack with no item size after pop", 0,
				stack.size());
		Testing.assertEquals("A stack with no item is empty after pop", true,
				stack.isEmpty());

	}
	
	private static void ConverterTester() {
		/*
		 *I wasn't sure the best way to test this, so I found an online infix-postfix converter and made a file in the format 
		 *of the output and I will use the file reader to put it into a string, then compare it to the actual output.
		 *
		 *So while its only one test, its testing a lot of different things
		 */
		FileReader correct = new FileReader("src/CorrectOutput.txt");
		Converter c = new Converter("src/input.txt");
		String correctString="";
		String tmp="";
		
		
		while(!tmp.equals("EOF")) { //While next is not EOF
			correctString += tmp;
			if(tmp.equals(";")) {
				correctString+="\n";
			}
			tmp=correct.nextToken();
		}
		
		Testing.assertEquals("Testing all of input file", correctString,
				c.convert());
		
	}
	
}