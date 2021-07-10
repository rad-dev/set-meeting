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