import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ThreeDSearchComponent } from './3d-search.component';

describe('ImageSearchComponent', () => {
  let component: ThreeDSearchComponent;
  let fixture: ComponentFixture<ThreeDSearchComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ThreeDSearchComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(ThreeDSearchComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
