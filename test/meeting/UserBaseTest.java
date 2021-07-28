package meeting;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;


import java.util.InvalidPropertiesFormatException;

public class UserBaseTest {
    private UserBase users;

    @BeforeEach
    void init() {
        users = new UserBase();
    }

    @AfterEach
    void teardown() {
        users = null;
    }

    @Test
    @DisplayName("Can add user")
    void canAddUser() throws InvalidPropertiesFormatException {
        Person newUser = new Person("Joe Fields", "j.fields@company.com");
        users.addUser(newUser);
    }
    @Test
    @DisplayName("Cannot add user when email already exists")
    void cannotAddUserWhenEmailExists() {
        Person userA = new Person("Joe Fields", "j.fields@company.com");
        Person userB = new Person("John Fields", "j.fields@company.com");

        try {
            users.addUser(userA);
        } catch (InvalidPropertiesFormatException e) {
            e.printStackTrace();
        }
        Assertions.assertThrows(InvalidPropertiesFormatException.class, () -> users.addUser(userB));
    }

    @Test
    @DisplayName("Can get user by email")
    void canGetUserByEmail() throws InvalidPropertiesFormatException {
        String email = "j.fields@company.com";
        Person userA = new Person("Joe Fields", email );
        users.addUser(userA);

       Assertions.assertEquals(userA, users.getUser(email));
    }
}