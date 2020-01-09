/**
 * Testing suite for BetterBag
 * 
 * @author Xavier Qunn, Chris Fernandes, and Matt Anderson
 * *I affirm that I have carried out the attached 
 *academic endeavors with full academic honesty, in
 *accordance with the Union College Honor Code and
 *the course syllabus.
 */
public class LinkedListTester {
    
    public static final boolean VERBOSE = true;
    
    /* Runs a bunch of tests for the BetterBag class.
     * @param args is ignored
     */
    public static void main(String[] args)
    {
    	
    Testing.setVerbose(true);
	Testing.startTests();
	
	testInserts();
	testRemove();
	
	
	Testing.finishTests();
	
    }
    

    
   
    
    private static void testInserts() {
    	Testing.testSection("Tests insertAtHead, insertAtTail, and toString");
    	LinkedList<String> list=new LinkedList<String>();
    	
    	LinkedList<String> list2=new LinkedList<String>();
    	
    	LinkedList<Integer> intList=new LinkedList<Integer>();

    	
    	
    	list.insertAt(0, "One");
    	Testing.assertEquals("Tests addition in empty list at start", "(One)", list.toString());
    	Testing.assertEquals("Tests addition in empty list capacity", 1, list.getLength());
    	
    	list.insertAt(5, "Two");
    	Testing.assertEquals("Tests addition at location longer than length", "(One, Two)", list.toString());
    	Testing.assertEquals("Tests addition in empty list capacity", 2, list.getLength());
    	
    	list.insertAt(1, "Three");
    	Testing.assertEquals("Tests addition between nodes", "(One, Three, Two)", list.toString());
    	Testing.assertEquals("Tests addition in empty list capacity", 3, list.getLength());
    	
    	list.insertAt(0, "Four");
    	Testing.assertEquals("Tests addition at start", "(Four, One, Three, Two)", list.toString());
    	Testing.assertEquals("Tests addition in empty list capacity", 4, list.getLength());
    	
    	list.insertAt(-6, "Five");
    	Testing.assertEquals("Tests addition at negative index", "(Five, Four, One, Three, Two)", list.toString());
    	
    	Testing.assertEquals("Tests addition in empty list capacity", 5, list.getLength());
    	list.insertAt(6, "Six");
    	Testing.assertEquals("Tests addition at end", "(Five, Four, One, Three, Two, Six)", list.toString());
    	Testing.assertEquals("Tests addition in empty list capacity", 6, list.getLength());
    	
    	list2.insertAt(0, "a");
    	list2.insertAt(1, null);
    	list2.insertAt(2, "b");


    	Testing.assertEquals("Tests addition between nodes", "(a, null, b)", list2.toString());
    	
    	
    	intList.insertAt(0, 1);
    	Testing.assertEquals("Tests addition in empty list at start", "(1)", intList.toString());
    	Testing.assertEquals("Tests addition in empty list capacity", 1, intList.getLength());
    	
    }
    
    
    private static void testRemove() {
    	Testing.testSection("Tests insertAtHead, insertAtTail, and toString");
    	LinkedList<String> list=new LinkedList<String>();
    	list.insertAt(10, "One");
    	list.insertAt(10, "Two");
    	list.insertAt(10, "Three");
    	list.insertAt(10, "Four");
    	Testing.assertEquals("Just checking", "(One, Two, Three, Four)", list.toString());
    	Testing.assertEquals("Tests addition in empty list capacity", 4, list.getLength());
    	
    	Testing.assertEquals("Test removal of last", "Four", list.removeAt(3));
    	Testing.assertEquals("Test removal of last", "(One, Two, Three)", list.toString());
    	Testing.assertEquals("Tests capacity after removal", 3, list.getLength());
    	
    	
    	Testing.assertEquals("Test removal of first", "One", list.removeAt(0));
    	Testing.assertEquals("Test removal of first", "(Two, Three)", list.toString());
    	Testing.assertEquals("Tests capacity after removal", 2, list.getLength());
    	
    	Testing.assertEquals("Test removal of first", null, list.removeAt(-5));
    	Testing.assertEquals("Test removal of first", "(Two, Three)", list.toString());
    	Testing.assertEquals("Tests capacity after removal", 2, list.getLength());


    	Testing.assertEquals("Test removal of first", null, list.removeAt(5));
    	Testing.assertEquals("Test removal of first", "(Two, Three)", list.toString());
    	Testing.assertEquals("Tests capacity after removal", 2, list.getLength());
    }
    
    
}