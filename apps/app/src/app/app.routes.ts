import { Routes } from '@angular/router';
import { AppLoginComponent } from './login/login.component';
import { AppComponent } from './app.component';
import { ThreeDSearchComponent } from './3d-search/3d-search.component';
import { AppRegisterComponent } from './register/register.component';
import { AppHomeComponent } from './home/home.component';

export const routes: Routes = [
  { path: '', redirectTo: 'home', pathMatch: 'full' },
  { path: 'home', component: AppHomeComponent },
  { path: 'login', component: AppLoginComponent },
  { path: 'register', component: AppRegisterComponent },
  { path: '3d-search', component: ThreeDSearchComponent },
];
