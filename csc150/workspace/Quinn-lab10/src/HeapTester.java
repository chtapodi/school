/**
 * Driver for the heap lab
 * 
 * 
 * 
 * @author <em>Xavier</em>, Chris Fernandes, and Aaron Cass
 * @version 3/13/13
 */
import java.util.Arrays;
public class HeapTester
{
    
    public static void main(String[] args)
    {
	
	Testing.startTests();
	Testing.setVerbose(true);
    	shallowHeap();
    	sortUnique();
    	testHeap();

	Testing.finishTests();

    }
    
    
    
    private static void testHeap() {
    	int[] arr=new int[3];
    	arr[0]=5;
    	arr[1]=20;
    	arr[2]=1;
    	Heap h = new Heap(arr);
    	
    	System.out.println(h.toString());
    	Testing.assertEquals("Removing a single node", 20, h.deleteRoot());
    	System.out.println(h.toString());
    	Testing.assertEquals("Removing a single node", 5, h.deleteRoot());
    	System.out.println(h.toString());
    	Testing.assertEquals("Removing a single node", 1, h.deleteRoot());
    	Testing.assertEquals("Removing a single node", 0, h.deleteRoot());

    	
    	//I cant't think of any other cases not covered by this or the other tests
    	
    }
    
    
    
    
    /**
     * Heap tests
     */
    private static void shallowHeap()
    {
    	Testing.testSection("heap test: subtree root swaps just once");
	
    	int[] anArray = {11, 12, 5, 1, 23, 33, 9, 21, 14, 10, 4};	
        printArray("before building heap:",anArray);
	
        Heap sample = new Heap(anArray);
		
    	Testing.assertEquals("shallowHeap: after building heap",
			     "33\n23 11\n21 12 5 9\n1 14 10 4", sample.toString());
    }
    
    /**
     * Sort tests
     */
    private static void sortUnique()
    {
	Testing.testSection("sort test: random, no duplicates");
	
    	int[] unsorted = {11, 12, 5, 1, 23, 33, 9, 21, 14, 10};

        int[] answer = {1, 5, 9, 10, 11, 12, 14, 21, 23, 33};
        
        
        //I dont know what was up with the code here, but It wouldnt give a passed, so I made a workaround.
        Testing.assertEquals("sortUnique: after sorting", arrString(answer), arrString(Sorter.priorityQueueSort(unsorted)));
    }


    /** 
     *  prints an array
     *  @param message string to print before printing array
     *  @param array the array of ints to be printed
     */
    private static void printArray(String message, int[] array) 
    {
	System.out.println(message);
	int len = array.length;
	for (int i = 0; i < len - 1; i++) {
	    System.out.print(array[i] + " ");
	}
	System.out.println(array[len-1] + "\n");
    }
    
    private static String arrString(int[] array) 
    {
    	String toReturn="";
    	int len = array.length;
    	for (int i = 0; i < len; i++) {
    		toReturn+= array[i] + " ";
    	}
    	return toReturn;
    }


}