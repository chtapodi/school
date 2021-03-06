/**
 * 
 * This is made to test all possible cases of the Sequence class by calling all methods in different situations.
 *  @author xavier
 *  
 *I affirm that I have carried out the attached 
 *academic endeavors with full academic honesty, in
 *accordance with the Union College Honor Code and
 *the course syllabus.
 */
public class SequenceTests {
    
    public static void main(String[] args)
    {
    	Testing.setVerbose(true); // use false for less testing output
		Testing.startTests();

    	testCreate();
		testAdding();
		testIsCurrent();
		testGetCurrent();
		testEnsureCapacity();
		testAddAll_Clone();
		testRemoveCurrent();
		testTrimToSize();
		testEquals();
		testIsEmpty();
		testClear();
	
    	// add calls to more test methods here.
    	// each of the test methods should be
    	// a private static method that tests
    	// one method in Sequence.
	
    	Testing.finishTests();
    	
    }
    
	private static void testCreate()
	{
		Testing.testSection("Creation tests and toString of empty sequence");
		
		Sequence s1 = new Sequence();
		Testing.assertEquals("Default constructor", "{} (capacity = 10)", s1.toString());
		Testing.assertEquals("Default constructor, initial size", 0, s1.size());
		
		Sequence s2 = new Sequence(20);
		Testing.assertEquals("Non-default constructor", "{} (capacity = 20)", s2.toString());
		Testing.assertEquals("Non-default constructor, initial size", 0, s2.size());
	}
	
	
	private static void testAdding() {
		Testing.testSection("Tests addBefore");
		
		Sequence s1 = new Sequence();
		Sequence s2 = new Sequence();
		s1.addBefore("one");
		Testing.assertEquals("Tests if added before works", "{>one} (capacity = 10)", s1.toString());
		
		s1.addAfter("two");
		Testing.assertEquals("Tests if added after works", "{one, >two} (capacity = 10)", s1.toString());
		Testing.assertEquals("Tests if added keeps current", "two", s1.getCurrent());
		
		s1.addBefore("three");
		Testing.assertEquals("Tests if added before works", "{one, >three, two} (capacity = 10)", s1.toString());
		Testing.assertEquals("Tests if added keeps current", "three", s1.getCurrent());
		
		s1.start();
		s1.addAfter("four");
		Testing.assertEquals("Tests if added after works with other parts", "{one, >four, three, two} (capacity = 10)", s1.toString());
		Testing.assertEquals("Tests if added keeps current", "four", s1.getCurrent());
		
		Testing.assertEquals("Tests if size is correct", 4, s1.size());
		
		
		s2.addAfter("test");
		Testing.assertEquals("Tests if added after works", "{>test} (capacity = 10)", s2.toString());
		
		
	}

	
	private static void testIsCurrent() {
		Testing.testSection("Tests isCurrent"); 
		
		Sequence s1 = new Sequence();
		Testing.assertEquals("Tests if current exists, doesn't", false, s1.isCurrent());
		
		s1.addAfter("tmp");
		Testing.assertEquals("Check if current exists, does", true, s1.isCurrent());
		
		
	}
	
	private static void testGetCurrent() {
		Testing.testSection("Tests getCurrent"); 
		
		Sequence s1 = new Sequence();
		Testing.assertEquals("Gets current value when it doesn't exist", null, s1.getCurrent());
		
		s1.addAfter("tmp");
		Testing.assertEquals("Gets current value", "tmp", s1.getCurrent());
		
	}

	private static void testEnsureCapacity() {
		Testing.testSection("Tests ensureCapacity"); 
		
		Sequence s1 = new Sequence();
		s1.ensureCapacity(5);
		Testing.assertEquals("Tests ensureCapacity when value is less than current", 10, s1.getCapacity());
		
		s1.ensureCapacity(15);
		Testing.assertEquals("Tests ensureCapacity when value is less than current", 15, s1.getCapacity());
		
		
	}
	
	private static void testAddAll_Clone() {
		Testing.testSection("Tests addAll and Clone"); 
		
		Sequence s1 = new Sequence(3);
		Sequence s2 = new Sequence(3);
		
		s1.addBefore("one");
		s1.addBefore("two");
		s1.addBefore("three");
		s2=s1.clone();
		
		
		s2=s1.clone();
		Testing.assertEquals("Tests clone", true, s1.equals(s2));
		
		s1.addBefore("four");
		
		Testing.assertEquals("Tests cloned sequence after one has been changed", false, s1.equals(s2));
		
		s1.addAll(s2);
		Testing.assertEquals("Tests addAll", "{>four, three, two, one, three, two} (capacity = 7)", s1.toString());
		
	}

	
	private static void testRemoveCurrent() {
		Testing.testSection("Tests removeCurrent"); 
		
		Sequence s1 = new Sequence();
		
		s1.removeCurrent();
		Testing.assertEquals("Tests removeCurrent with empty sequence", "{} (capacity = 10)", s1.toString());
		
		s1.addAfter("one");
		s1.addAfter("two");
		
		s1.removeCurrent();
		Testing.assertEquals("Tests removeCurrent at end of sequence", "{one} (capacity = 10)", s1.toString());
		Testing.assertEquals("Tests checks for current value (doesnt exist)", null, s1.getCurrent());
		s1.removeCurrent();
		Testing.assertEquals("Tests removeCurrent", null, s1.getCurrent());
		
		s1.addAfter("three");
		s1.addAfter("four");
		s1.start();
		s1.removeCurrent();
		Testing.assertEquals("Tests removeCurrent with values after it", "{>three, four} (capacity = 10)", s1.toString());
		Testing.assertEquals("Tests checks for current value", "three", s1.getCurrent());
		
	}
	
	
	private static void testTrimToSize() {
		Testing.testSection("Tests trimToSize"); 
		
		Sequence s1 = new Sequence();
		
		s1.addAfter("one");
		s1.addBefore("two");
		s1.trimToSize();
		Testing.assertEquals("Tests trim to size of 2", 2, s1.getCapacity());
		
		
	}
	
	private static void testEquals() {
		Testing.testSection("Tests equals"); 
		
		Sequence s1 = new Sequence();
		Sequence s2 = new Sequence();
		
		s1.addAfter("tmp");
		s1.addBefore("first");
		s2 = s1.clone();
		Testing.assertEquals("Tests equals, should be true", true, s1.equals(s2));
		
		s1.addAfter("fred");
		Testing.assertEquals("Tests equals, should be false", false, s1.equals(s2));
		
		
		
		
	}
	
	private static void testIsEmpty() {
		Testing.testSection("Tests isEmpty"); 
		
		Sequence s1 = new Sequence();
		Testing.assertEquals("Tests if empty Sequence is empty", true, s1.isEmpty());
		
		s1.addAfter("tmp");
		Testing.assertEquals("Tests if non-empty Sequence is empty", false, s1.isEmpty());
		
	}
	
	private static void testClear() {
		Testing.testSection("Tests clear"); 
		
		Sequence s1 = new Sequence();
		s1.addAfter("tmp");
		
		
	}
}
