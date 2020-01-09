/**
 * An extremely scaled-down version of a heap-based PQ.  A "real" PQ would have a default
 * constructor to create an empty PQ and an insert method.  This PQ only 
 * handles ints.
 * 
 * @author Chris Fernandes
 * @version 3/13/13
 */

public class PriorityQueue
{
    private Heap PQHeap;    // the heap that holds PQ items
     
    /**
     * Constructs a priority queue with given initial values.
     *
     * @param initialValues the array of values contained in the priority queue
     */
    public PriorityQueue(int[] initialValues) 
    {    
      PQHeap = new Heap(initialValues);
    }                                                 

    /**
     * @return number of ints in the PriorityQueue.
     */
    public int size() 
    {
      return PQHeap.size();
    }
    
    /**
     * Removes and returns the largest element in the PriorityQueue.
     *
     * @return the largest element.
     */
    public int remove() 
    {
        return PQHeap.deleteRoot();
    }
}