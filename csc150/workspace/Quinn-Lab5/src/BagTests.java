/**
 * Testing suite for BetterBag
 * 
 * @author Xavier Qunn, Chris Fernandes, and Matt Anderson
 * *I affirm that I have carried out the attached 
 *academic endeavors with full academic honesty, in
 *accordance with the Union College Honor Code and
 *the course syllabus.
 */
public class BagTests {
    
    public static final boolean VERBOSE = true;
    
    /* Runs a bunch of tests for the BetterBag class.
     * @param args is ignored
     */
    public static void main(String[] args)
    {
	Testing.startTests();
	
	testClone();
	//uncomment these when you are ready to test them
	testisEmpty();
	testRemoveRandom();
	testContains();
	testEquals();
	
	Testing.finishTests();
	
    }
    
    private static void testClone()
    {
	Testing.testSection("Testing clone()");
	
	BetterBag bag1 = new BetterBag(3);
	
	BetterBag bag2 = bag1.clone();
	Testing.assertEquals("cloning an empty sequence", "{} (capacity = 3)", bag2.toString());
	
	bag1 = new BetterBag(3);
	bag1.add(4);
	bag1.add(8);
	bag1.add(12);
	bag2 = bag1.clone();
	Testing.assertEquals("cloning {4, 8, 12}", "{4, 8, 12} (capacity = 3)", bag2.toString());
	Testing.assertEquals("cloning {4, 8, 12} should produce a different object.  Does (bag2 != bag1)",
			     true, (bag2 != bag1));
	
	bag1 = new BetterBag(7);
	bag1.add(-1);
	bag1.add(-2);
	bag1.add(-3);
	bag2 = bag1.clone();
	bag1.add(-4);
	Testing.assertEquals("clone shouldn't change after adding to original",
			     "{-1, -2, -3} (capacity = 7)", bag2.toString());
	Testing.assertEquals("original should change after cloning & adding to original", 
			     "{-1, -2, -3, -4} (capacity = 7)", bag1.toString());
	
	bag1 = new BetterBag(5);
	bag1.add(1);
	bag1.add(2);
	bag1.add(3);
	bag1.add(4);
	bag2 = bag1.clone();
	bag2.add(5);
	Testing.assertEquals("original shouldn't change after adding to clone",
			     "{1, 2, 3, 4} (capacity = 5)", bag1.toString());
	Testing.assertEquals("clone should change after cloning & adding to clone", 
			     "{1, 2, 3, 4, 5} (capacity = 5)", bag2.toString());
    }
    
    
    
    private static void testisEmpty() {
    	Testing.testSection("Testing isEmpty()");
    	BetterBag bag1;
    	bag1=bagInit();
    	
    	Testing.assertEquals("Testing if full bag is empty. Should be false", false, bag1.isEmpty());
    	
    	BetterBag bag2= new BetterBag(3);
    	Testing.assertEquals("Testing if empty bag is empty. Should be true", true, bag2.isEmpty());
    }
    
    
    private static void testRemoveRandom() {
    	Testing.testSection("Testing removeRandom()");
    	BetterBag bag1;
    	BetterBag bagSave;
    	bag1=bagInit();
    	bagSave=bag1.clone();
    	
      	Testing.assertEquals("The bag after removing a random value should be false", true, (bagSave != bag1));
    	
    	BetterBag bag2= new BetterBag(3);
    	Testing.assertEquals("Testing if empty bag is empty. Should be true", Integer.MIN_VALUE, bag2.removeRandom());
    }
    
    
    private static void testContains() {
    	Testing.testSection("Testing contains()");
    	BetterBag bag1;
    	bag1=bagInit();
    	BetterBag bag2 = new BetterBag(0);
    	
    	
    	Testing.assertEquals("Testing if bag contains 5. Should be false", false, bag1.contains(5));
    	Testing.assertEquals("Testing if bag contains 0. Should be true", true, bag1.contains(0));
    	
    	
    	Testing.assertEquals("Testing if empty bag contains 5. Should be false", false, bag2.contains(5));
    }
    
    
    private static void testEquals() {
    	Testing.testSection("Testing equals()");
    	BetterBag bag1;
    	bag1=bagInit();
    	BetterBag sameBag = new BetterBag(5);
    	sameBag.add(1);
    	sameBag.add(-2);
    	sameBag.add(0);
    	sameBag.add(600);
    	
    	
    	BetterBag diffBag = new BetterBag(3);
    	diffBag.add(0);
    	diffBag.add(-3);
    	diffBag.add(700);
    	diffBag.add(1);
    	

    	Testing.assertEquals("Testing if equal bags are equal. Should be false", true, bag1.equals(sameBag));
    	Testing.assertEquals("Testing if bag containing nothing is equal. Should be false", false, bag1.equals(diffBag));
    	
    	
    }
    
    
    
    private static BetterBag bagInit() {
    	BetterBag bag= new BetterBag(10);
    	BetterBag bagSave;
    	//Arbitrary values
    	bag.add(1);
    	bag.add(-2);
    	bag.add(600);
    	bag.add(0);
    	return bag;
    }
    
    
    
}