/**
 * Describes the methods that must be defined to use be a LogEntry
 * 
 * @author Xavier Quinn
 * @version 2/23/17
 *
 */
public interface LogEntry
{
	
	/**
	 * Returns the contents as a string
	 * @return The string of the contents
	 */
	public String toString();
	
	/**
	 * Returns the month as an int
	 * @return the month (1-12)
	 */
	public int getMonth();
	
	/**
	 * Returns the day as an int
	 * @return The day (1-31)
	 */
	public int getDay();
	
	/**
	 * Returns the year as an int
	 * @return The year (e.g. 2017)
	 */
	public int getYear();
	
	
}
