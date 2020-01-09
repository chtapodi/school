

/**
 * An ADT that holds a single string and a list of page numbers that the string occurs on.
 * There isn't really a use to this besides making an index for a document
 * 
 * There is a preset limit to the number of pages this will hold before returning that it is full.
 * 
 * @author xavier
 * 
 */
public class Document implements Comparable<Document> {

	private String word;
	private LinkedList<Integer> pages;
	private final int MAX_PAGES=5;
	
	public Document(String toStore, int page) {
		word=toStore;
		
		pages=new LinkedList<Integer>();
		pages.insertAt(pages.getLength(), page);
	}
	
	/**
	 * Adds an instance of the word, returns true if the instance has reached the limit
	 * There will never be duplicate pages
	 * @param page The page number
	 * @return True or false based on if the max of this word has been reached
	 */
	public boolean addInstance(int page) {
		
		if(pages.getData(pages.getLength()-1)!=page) { //If most recent page isn't the same as the new page, because they will be inserted in order
			
			pages.insertAt(pages.getLength(), page);
			
			if(pages.getLength()>=MAX_PAGES) {
				return true;
			}
		}

		return false; //I tried putting this in an else{}, but eclipse wouldn't allow it for some reason

		
		
	}
	
	/**
	 * Returns the word of this document
	 * @return The word
	 */
	public String getWord() {
		return this.word.toLowerCase();
	}
	
	/**
	 * Compares the words of the two documents
	 * @param other The other document to be compared to 
	 * @return <0 if smaller than other, 0 if other, >0 if greater
	 */
	public int compareTo(Document other) {
		return this.word.compareToIgnoreCase(other.getWord());
	}
	
	
	
	/**
	 * Compares the word of this document to a string
	 * @param other The String to be compared to 
	 * @return <0 if smaller than other, 0 if other, >0 if greater
	 */
	public int compareTo(String other) {
		return this.word.compareToIgnoreCase(other);
	}
	
	
	/**
	 * Counts how many pages the word appears consecutively on
	 * @param index The page to start on
	 * @return The number of sequencial pages it apears on
	 */
	private int sequenceCheck(int index) {
		if(pages.getData(index+1)!=null && pages.getData(index)+1==pages.getData(index+1)) {
			return sequenceCheck(index+1) + 1;
		}
		else {
			return 0;
		}
	}
	
	/**
	 * returns the word and the pages it appears on, if there are sequential pages, it shows them as e.g 1-3
	 * @return The word and the pages it appears on
	 */
	public String toString() {
		String toReturn="";
		int length=pages.getLength();
		int start=pages.getData(0);
		int seq=0;
		
		
		
		for(int i=0;i<length;i++) {
			seq=sequenceCheck(i);

			if(seq>1) {
				toReturn+=pages.getData(i)+"-"+((int)pages.getData(i)+(int)seq)+", ";
				i=i+seq;
			}
			else {
				toReturn+=pages.getData(i)+", ";
			}
		}
		
		return this.word.toLowerCase()  + ": " + toReturn.substring(0, toReturn.length() - 2);
	}

}
