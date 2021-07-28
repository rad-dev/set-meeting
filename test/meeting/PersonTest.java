package meeting;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class PersonTest {

    @Test
    @DisplayName("Person can set name")
    void canSetName() {
        String name = "Joe Smith";
        person.setName(name);
        assertEquals(name, person.getName());
    }

    @Test
    @DisplayName("Person can set email")
    void canSetEmail() {
        String email = "joe.smith@company.com";
        person.setEmail(email);
        assertEquals(email, person.getEmail());
    }

    @Test
    @DisplayName("Can set meeting in the morning [time = 10]")
    void canSetMorningMeetings () {
        int time = 10;
        assertTrue(person.setMeeting(time));
    }

    @Test
    @DisplayName("Can print meetings")
    void canPrintMeetings(){
        int time1 = 10;
        int time2 = 3;
        int time3 = 5;
        this.person.setMeeting(time1);
        this.person.setMeeting(time2);
        this.person.setMeeting(time3);

        String meetings = this.person.printMeetings();
        System.out.println(meetings);
    }

    @Test

    @AfterEach
    void tearDown(){
        this.person = null;
    }

    @BeforeEach
    void init() {
        this.person = new Person("", "");
    }

    Person person;
}