/**
 * Given an input text, goes though and indexs the words in the text. It does not index words that are too common, and instead places them in a dictionary
 * It also does not document or index words shorter than three letters
 * 
 */

/**
 * @author xavier
 * 
 */
public class Indexer {
	
	private FileReader r;
	private BinarySearchTree dictionary;
	private BinarySearchTree index;
	private final int MIN_WORD_LENGTH=3;
	
	
	private String removed;
	
	/**
	 * Constructor for the indexer, which takes the source file as the input
	 * @param text The source file.
	 */
	public Indexer(String text) {
		r=new FileReader(text);
		dictionary=new BinarySearchTree<String>();
		index=new BinarySearchTree<Document>();
		
		removed="";
	}
	

	
	
	
	
	
	/**
	 * Reads through the text and sorts the words in the appropriate manor
	 */
	public void interpret() {
		int pageNumber=1;
		String tmp = r.nextToken();
		
		while(tmp!=null) {
			if(tmp.equals("#")) {
				pageNumber++;
			}
			else if (tmp.length()>=MIN_WORD_LENGTH && searchDictionary(tmp)==null) {
				toIndex(tmp, pageNumber);
			}
			tmp=r.nextToken();
		}
		System.out.println("Dictionary:\n" + dictionary.sortedString());
		System.out.println("Index:\n" + index.sortedString());
	}
	
	
	/**
	 * Returns the Document with the given word, or null if not found
	 * @param word What to search the dictionary for
	 * @return The document containing the word, or null
	 */
	private String searchDictionary(String word) {
		return (String)dictionary.search(word);
	}
	
	
	/**
	 * Returns the Document with the given word, or null if not found
	 * @param word What to search the index for
	 * @return The document containing the word, or null
	 */
	private Document searchIndex(String word) {
		Document forSearch = new Document(word, 0);
		return (Document)index.search(forSearch);
	}
	
	
	/**
	 * Handles checking if the data is already in the index, and if not inserting it there.
	 * @param input
	 * @param pageNumber
	 */
	private void toIndex(String input, int pageNumber) {
		Document tmpDoc=searchIndex(input);
		if(tmpDoc!=null) { //if already in index
			
			if(tmpDoc.addInstance(pageNumber)) { //If the pagelist is full
				System.out.println("removed " + tmpDoc.toString());
				index.remove(tmpDoc);
				removed +=input+"\n";
				dictionary.insert(input);
			}
		}
		else {
			
			insert(input, pageNumber);
		}
	}
	
	
	/**
	 * Inserts a word with the given stuff to the index
	 * @param word The word
	 * @param pageNumber The pagenumber
	 */
	private void insert(String word, int pageNumber) {
		Document toInsert=new Document(word, pageNumber);
		index.insert(toInsert);
	}
	
	

	

}
