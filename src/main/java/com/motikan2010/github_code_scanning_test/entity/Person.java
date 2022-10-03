package com.motikan2010.github_code_scanning_test.entity;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;
import java.io.Serializable;

@Entity
@Table(name = "person")
public class Person implements Serializable {

    @Id
    @Column(name="id")
    private Integer id;

    @Column(name="name")
    private String name;

    @Column(name="age")
    private Integer age;

    public Integer getId() {
        return this.id;
    }

    public String getName() {
        return this.name;
    }

    public Integer getAge() {
        return this.age;
    }

    @Override
    public String toString() {
        return "Person [id=" + this.id + ", name=" + this.name + ", age=" + this.age + "]";
    }

}
