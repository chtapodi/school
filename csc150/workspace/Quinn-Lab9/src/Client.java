/** Simply runs tests on the bianary search tree
 * 
 * 
 * * I affirm that I have carried out the attached academic endeavors
 *          with full academic honesty, in accordance with the Union College
 *          Honor Code and the course syllabus.
 * 
 * @author  Xavier
 * @version 3/2/2017
 */
public class Client
{
    public static void main(String[] args)
    {
	Testing.startTests();

	// Put tests here!
	testInsertion();
	testSearch();
	
	Testing.finishTests();
    }
    
    
    
    private static void testInsertion() {
    	
    	BinarySearchTree tree = new BinarySearchTree<Integer>();
    	
    	for(int i=5;i<15;i+=3) {
    		tree.insert(i);
    	}
    	Testing.assertEquals("Testing the addition of 4 values in order", "(  5 (  8 (  11 (  14 ))))", tree.toString());
    	
    	
    	for(int i=4;i<16;i+=3) {
    		tree.insert(i);
    	}
    	
    	Testing.assertEquals("Testing the addition of 4 more values out of order", "((  4 )  5 ((  7 )  8 ((  10 )  11 ((  13 )  14 ))))", tree.toString());
    	
    	for(int i=5;i<15;i+=2) {
    		tree.insert(i);
    	}
    	
    	Testing.assertEquals("Testing the addition of 5 more values out of order", "((  4 (  5 ))  5 (((  7 )  7 )  8 (((  9 )  10 (  11 ))  11 (((  13 )  13 )  14 ))))", tree.toString());
    	
    	
    }
    
    private static void testSearch() {
    	BinarySearchTree tree = new BinarySearchTree<Integer>();

    	for(int i=5;i<15;i+=3) {
    		tree.insert(i);
    	}
    	for(int i=4;i<16;i+=3) {
    		tree.insert(i);
    	}
    	
//    	Testing.assertEquals("Searches for a number that exists", 10, tree.search(10).key);
//    	Testing.assertEquals("Searches for a number that exists", 13, tree.search(13).key);
//    	Testing.assertEquals("Searches for a number that exists", 8, tree.search(8).key);
    	
    	//Testing.assertEquals("I can't think of a good way to test this without throwing a null pointer exception.", null, tree.search(0).toString());
    	//search should return null, so I can't get the info of it without getting an error.
    }

}