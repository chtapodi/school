/**
 * Improved bag of integers.  A bag is an unordered collection of elements.
 * Duplicates are allowed.
 * 
 * @author <em>Xavier Quinn</em> and Chris Fernandes
 * @version <em>feb 02 107</em>
 *
 */


public class BetterBag {

	private IntBag intBag;
		
	/**
	 * Initialize an empty bag with an initial capacity of 10.
	 */
	public BetterBag() {
		intBag = new IntBag();
	}
	
	/**
	 * Initialize an empty bag with a specified initial capacity.
	 * @param initialCapacity  the initial capacity of this bag
	 */
	public BetterBag(int initialCapacity) {
		intBag = new IntBag(initialCapacity);
	}
	
	/**
	 * Add a new element to this bag. If the new element would take this
	 * bag beyond its current capacity, then the capacity is increased
	 * before adding the new element.
	 * @param element the new element that is being inserted
	 */
	public void add(int element) {
		intBag.add(element);
	}
	
	/**
	 * Add the contents of another bag to this bag
	 * @param addend a bag whose contents will be added to this bag
	 */
	public void addAll(BetterBag addend) {
		intBag.addAll(addend.intBag);
	}
	
	/**
	 * Generate a copy of this bag.
	 * @return returns a copy of this bag. Subsequent changes to the
	 *   copy will not affect the original, nor vice versa.
	 */
	public BetterBag clone() {
		IntBag newInnerBag = intBag.clone();
		BetterBag toReturn = new BetterBag();
		toReturn.intBag = newInnerBag;
		return toReturn;
	}
	
	/**
	 * Counts the number of occurrences of a particular element
	 * in this bag.
	 * @param target
	 *   the element that needs to be counted
	 * @return
	 *   the number of times that target occurs in this bag
	 */
	public int countOccurrences(int target) {
		return intBag.countOccurrences(target);
	}
	
	/**
	 * Change the capacity of this bag to at least minimumCapacity.
	 *   If the capacity was already at or greater than minimumCapacity,
	 *   then the capacity is left unchanged.
	 *   
	 * @param minimumCapacity
	 *   the new capacity for this bag
	 */
	public void ensureCapacity(int minimumCapacity) {
		intBag.ensureCapacity(minimumCapacity);
	}
	
	/**
	 * Getter for this bag's capacity
	 * @return the number of elements this bag has the ability to hold
	 */
	public int getCapacity() {
		return intBag.getCapacity();
	}
	
	/**
	 * Remove one copy of a specified element from this bag.
	 * @param target the element to remove from the bag
	 * @return true if removal was successful; false if target doesn't exist
	 *  (which means the bag will be left unchanged)
	 */
	public boolean remove(int target) {
		return intBag.remove(target);
	}
	
	/**
	 * Determine the number of elements in this bag
	 * @return the number of elements currently in this bag
	 */
	public int size() {
		return intBag.size();
	}
	
	/**
	 * Reduce the current capacity of this bag to its actual size (i.e., the
	 * number of elements it contains).
	 */
	public void trimToSize() {
		intBag.trimToSize();
	}
	
	/**
	 * Accesses a random element in this bag
	 * 
	 * @return a random element from this bag.  The bag is left unchanged
	 *  (i.e. the element is not removed).  If the bag is empty,
	 *  the smallest possible integer is returned (i.e. the static
	 *  constant <code>Integer.MIN_VALUE</code> is returned)
	 */
	public int getRandom() {
		return intBag.getRandom();
	}
	
	/**
	 * Return status of this bag as a String.  For example, if a bag of 
	 * capacity 10 contained the elements 1, 2, and 3, the returned
	 * string would be "{1, 2, 3} (capacity = 10)".  The order in which
	 * the elements are listed is not guaranteed.
	 * 
	 * @return bag's contents and capacity as a String 
	 */
	public String toString() {
		return intBag.toString();
	}
	
	/**
	 * Tells if bag is empty or not
	 * @return true if bag contains no elements; false otherwise
	 */
	public boolean isEmpty() {
		
		if(this.size()>0) {
			 return false;
		}
		
		return true;
	}

	/**
	 * remove a random element from this bag and return it
	 * 
	 * @return the random element that has been removed from
	 *  this bag.  If the bag is empty, it is left unchanged and
	 *  the smallest possible integer is returned (i.e. the static
	 *  constant <code>Integer.MIN_VALUE</code> is returned)
	 */
	public int removeRandom() {
		int tmpInt=this.getRandom();
		this.remove(tmpInt);
		return tmpInt;
	}
	
	/**
	 * See if the bag contains a particular element
	 * @param target the element to be searched for
	 * @return true if bag contains the target element; false otherwise
	 */
	public boolean contains(int target) {
		if (this.countOccurrences(target)>0) {
			return true;
		}
		
		return false;
	}
	
	/**
	 * Tests to see if this bag and another bag are equal. Two bags
	 * are equal if they contain the exact same elements, regardless of order
	 * or capacity.  For example, the bags {1,4,9} and {9,4,1} are equal,
	 * even if they had different capacities.  The number of duplicate
	 * elements must also be the same, so
	 * {1,4,4,9} and {1,4,9,9} are not equal. 
	 * 
	 * @param otherBag bag to be tested for equality with this bag
	 * @return true if bags are equal; false otherwise
	 */
	public boolean equals(BetterBag otherBag) {
		BetterBag copyBag;
		BetterBag copyOtherBag;
		copyBag=this.clone();
		copyOtherBag=otherBag.clone();
		
		for(int i=0;i<copyBag.getCapacity();i++) {
			int tmpVal=copyBag.getRandom();
			
			if(copyBag.countOccurrences(tmpVal)!=copyOtherBag.countOccurrences(tmpVal)) {
				return false;
			}
			copyBag.remove(tmpVal);
			copyOtherBag.remove(tmpVal);
		}
		
    	return true;
	}
}