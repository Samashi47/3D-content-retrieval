import {
  ChangeDetectionStrategy,
  Component,
  CUSTOM_ELEMENTS_SCHEMA,
  inject,
  OnInit,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import {
  FormControl,
  FormsModule,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatSnackBar } from '@angular/material/snack-bar';
import { merge } from 'rxjs';
import { AuthService } from '../auth.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrl: './login.component.css',
  imports: [
    MatFormFieldModule,
    MatIconModule,
    MatButtonModule,
    MatInputModule,
    FormsModule,
    ReactiveFormsModule,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
})
export class AppLoginComponent implements OnInit {
  private _snackBar = inject(MatSnackBar);

  openSnackBar(message: string, action: string) {
    return this._snackBar.open(message, action, {
      duration: 2000,
    });
  }

  readonly email = new FormControl('', [Validators.required, Validators.email]);

  errorEmailMessage = signal('');

  constructor(private authService: AuthService, private router: Router) {
    merge(this.email.statusChanges, this.email.valueChanges)
      .pipe(takeUntilDestroyed())
      .subscribe(() => this.updateEmailErrorMessage());

    merge(this.password.statusChanges, this.password.valueChanges)
      .pipe(takeUntilDestroyed())
      .subscribe(() => this.updatePasswordErrorMessage());
  }

  ngOnInit() {
    if (this.authService.isLoggedIn()) {
      this.router.navigate(['/3d-search']);
    }
  }

  updateEmailErrorMessage() {
    if (this.email.hasError('required')) {
      this.errorEmailMessage.set('You must enter a value');
    } else if (this.email.hasError('email')) {
      this.errorEmailMessage.set('Not a valid email');
    } else {
      this.errorEmailMessage.set('');
    }
  }

  readonly password = new FormControl('', [
    Validators.required,
    Validators.minLength(8),
  ]);

  errorPasswordMessage = signal('');

  updatePasswordErrorMessage() {
    if (this.password.hasError('required')) {
      this.errorPasswordMessage.set('You must enter a value');
    } else if (this.password.hasError('minlength')) {
      this.errorPasswordMessage.set('Password must be at least 8 characters');
    } else {
      this.errorPasswordMessage.set('');
    }
  }

  hide = signal(true);
  clickEvent(event: MouseEvent) {
    this.hide.set(!this.hide());
    event.stopPropagation();
  }

  submit() {
    if (this.email.valid && this.password.valid) {
      console.log('Form submitted');
      const email = this.email.value ?? '';
      const password = this.password.value ?? '';
      const response = this.authService.login(email, password);

      response.subscribe(
        (data: any) => {
          this.authService.setToken(data.token);
          if (this.authService.isLoggedIn()) {
            let snackBarRef = this.openSnackBar('Login successful', 'Dismiss');
            snackBarRef.afterDismissed().subscribe(() => {
              this.router.navigate(['/3d-search']);
            });
            snackBarRef.onAction().subscribe(() => {
              this.router.navigate(['/3d-search']);
            });
          }
        },
        (error) => {
          let snackBarRef = this.openSnackBar('Invalid credentials', 'Dismiss');
        }
      );
    } else {
      let snackBarRef = this.openSnackBar('Invalid form', 'Dismiss');
    }
  }
}
